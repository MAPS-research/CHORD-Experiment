from __future__ import annotations

import math
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

import numpy as np

from chord.utils.hashing import sha256_text, stable_hash


def _cache_connection(path: str | Path) -> sqlite3.Connection:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(destination)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    return connection


def _chunks(values: Sequence[str], size: int = 400) -> Iterable[Sequence[str]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


class GenerationPerplexityScorer:
    """Token-weighted causal-LM perplexity with a persistent per-text cache."""

    def __init__(self, config: Dict[str, Any]) -> None:
        try:
            import torch
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("gen-PPL requires the 'hf' extra") from exc

        self.torch = torch
        self.config = dict(config)
        self.model_name = str(config["model"])
        self.revision = config.get("revision")
        self.tokenizer_revision = config.get("tokenizer_revision") or self.revision
        self.max_length = int(config.get("max_length", 512))
        self.batch_size = int(config.get("batch_size", 4))
        self.device = torch.device(
            config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        )
        cache_dir = config.get("hf_cache_dir")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            revision=self.tokenizer_revision,
            cache_dir=cache_dir,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = None
        self.protocol_hash = stable_hash(
            {
                "model": self.model_name,
                "revision": self.revision,
                "tokenizer_revision": self.tokenizer_revision,
                "max_length": self.max_length,
            }
        )
        self.connection = _cache_connection(config["cache_path"])
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS generation_scores (
                protocol_hash TEXT NOT NULL,
                text_hash TEXT NOT NULL,
                total_nll REAL,
                token_count INTEGER NOT NULL,
                token_ids BLOB NOT NULL,
                PRIMARY KEY (protocol_hash, text_hash)
            )
            """
        )

    def _ensure_model(self) -> Any:
        if self.model is not None:
            return self.model
        from transformers import AutoModelForCausalLM

        kwargs: Dict[str, Any] = {
            "revision": self.revision,
            "cache_dir": self.config.get("hf_cache_dir"),
        }
        dtype_name = self.config.get("model_dtype")
        if dtype_name:
            kwargs["torch_dtype"] = getattr(self.torch, dtype_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **kwargs)
        self.model.to(self.device).eval()
        return self.model

    def _cached_rows(self, hashes: Sequence[str]) -> Dict[str, tuple]:
        output: Dict[str, tuple] = {}
        for chunk in _chunks(list(dict.fromkeys(hashes))):
            placeholders = ",".join("?" for _ in chunk)
            query = (
                "SELECT text_hash, total_nll, token_count, token_ids "
                "FROM generation_scores WHERE protocol_hash = ? "
                f"AND text_hash IN ({placeholders})"
            )
            for row in self.connection.execute(query, (self.protocol_hash, *chunk)):
                output[str(row[0])] = (row[1], int(row[2]), row[3])
        return output

    def _tokenize_and_store(self, texts: Sequence[str]) -> Dict[str, tuple]:
        hashes = [sha256_text(text) for text in texts]
        cached = self._cached_rows(hashes)
        missing = [
            (text_hash, text)
            for text_hash, text in dict(zip(hashes, texts)).items()
            if text_hash not in cached
        ]
        for start in range(0, len(missing), self.batch_size):
            batch = missing[start : start + self.batch_size]
            tokenized = self.tokenizer(
                [text for _, text in batch],
                padding=False,
                truncation=True,
                max_length=self.max_length,
                add_special_tokens=True,
            )
            rows = []
            for (text_hash, _), token_ids in zip(batch, tokenized["input_ids"]):
                values = np.asarray(token_ids, dtype=np.int32)
                token_count = max(len(values) - 1, 0)
                blob = values.tobytes()
                rows.append((self.protocol_hash, text_hash, None, token_count, blob))
                cached[text_hash] = (None, token_count, blob)
            self.connection.executemany(
                """
                INSERT OR REPLACE INTO generation_scores
                (protocol_hash, text_hash, total_nll, token_count, token_ids)
                VALUES (?, ?, ?, ?, ?)
                """,
                rows,
            )
        self.connection.commit()
        return cached

    def _score_missing(self, texts: Sequence[str], cached: Dict[str, tuple]) -> None:
        missing = [
            (sha256_text(text), text)
            for text in dict.fromkeys(texts)
            if cached[sha256_text(text)][0] is None
        ]
        if not missing:
            return
        model = self._ensure_model()
        torch = self.torch
        for start in range(0, len(missing), self.batch_size):
            batch = missing[start : start + self.batch_size]
            encoded = self.tokenizer(
                [text for _, text in batch],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                add_special_tokens=True,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)
            with torch.inference_mode():
                logits = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                ).logits
            shift_logits = logits[:, :-1, :].float()
            shift_labels = input_ids[:, 1:]
            shift_mask = attention_mask[:, 1:].bool()
            losses = torch.nn.functional.cross_entropy(
                shift_logits.transpose(1, 2),
                shift_labels,
                reduction="none",
            )
            rows = []
            for index, (text_hash, _) in enumerate(batch):
                total_nll = float(losses[index][shift_mask[index]].sum().item())
                token_ids = np.asarray(
                    encoded["input_ids"][index][encoded["attention_mask"][index].bool()]
                    .cpu()
                    .numpy(),
                    dtype=np.int32,
                )
                token_count = max(len(token_ids) - 1, 0)
                blob = token_ids.tobytes()
                rows.append(
                    (
                        total_nll,
                        token_count,
                        blob,
                        self.protocol_hash,
                        text_hash,
                    )
                )
                cached[text_hash] = (total_nll, token_count, blob)
            self.connection.executemany(
                """
                UPDATE generation_scores
                SET total_nll = ?, token_count = ?, token_ids = ?
                WHERE protocol_hash = ? AND text_hash = ?
                """,
                rows,
            )
            self.connection.commit()

    def perplexity(self, texts: Sequence[str]) -> float:
        cached = self._tokenize_and_store(texts)
        self._score_missing(texts, cached)
        total_nll = 0.0
        total_tokens = 0
        for text in texts:
            nll, token_count, _ = cached[sha256_text(text)]
            if nll is not None:
                total_nll += float(nll)
                total_tokens += int(token_count)
        if total_tokens == 0:
            raise ValueError("no predicted tokens available for perplexity scoring")
        return float(math.exp(total_nll / total_tokens))

    def unigram_entropy(self, texts: Sequence[str]) -> float:
        cached = self._tokenize_and_store(texts)
        counts: Counter[int] = Counter()
        for text in texts:
            _, _, blob = cached[sha256_text(text)]
            token_ids = np.frombuffer(blob, dtype=np.int32)
            counts.update(int(value) for value in token_ids[1:])
        total = sum(counts.values())
        if not total:
            return 0.0
        return -sum((count / total) * math.log(count / total) for count in counts.values())
