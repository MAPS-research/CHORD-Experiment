"""Prefix-continuation generators for Figure 5 (GPT-2 / MDLM / SEDD) from a prefixes file.

Each model continues from the SAME prefixes (built by build_continuation_corpus.py) so the
continuations are directly comparable to the human continuation under any metric.

- AR (GPT-2): HF generate from the prefix; decoding in {nucleus, greedy, hot}.
- MDLM: masked-diffusion infill — initialize x with the prefix tokens (the loop already
  freezes any non-mask position), only the suffix starts masked.
- SEDD: official analytic PC sampler with a proj_fun that clamps the prefix positions
  every step (mirrors the repo's run_sample_cond.py).

Writes {"id","text"} JSONL where text = the CONTINUATION ONLY (suffix), matching how
TextLDM/MAUVE compare generated targets to the ground-truth target.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_prefixes(path):
    return [json.loads(line) for line in open(path)]


def run_ar(args, prefixes):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model, cache_dir=args.cache_dir, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = (
        AutoModelForCausalLM.from_pretrained(args.model, cache_dir=args.cache_dir)
        .to(args.device)
        .eval()
    )
    kw = dict(
        max_new_tokens=args.cont_len, min_new_tokens=args.cont_len, pad_token_id=tok.eos_token_id
    )
    if args.decoding == "nucleus":
        kw.update(do_sample=True, top_p=0.95, temperature=1.0)
    elif args.decoding == "greedy":
        kw.update(do_sample=False)
    elif args.decoding == "hot":
        kw.update(do_sample=True, temperature=1.6, top_p=1.0)
    else:
        raise ValueError(args.decoding)
    out = []
    for i in range(0, len(prefixes), args.batch_size):
        chunk = prefixes[i : i + args.batch_size]
        enc = tok([p["prefix_text"] for p in chunk], return_tensors="pt", padding=True).to(
            args.device
        )
        torch.manual_seed(args.seed + i)
        with torch.no_grad():
            o = model.generate(**enc, **kw)
        new = o[:, enc["input_ids"].shape[1] :]
        for p, row in zip(chunk, new):
            # same whitespace normalization as the MDLM/SEDD lanes: the reference pool
            # and human continuations are newline-free (upstream corpus prep flattens
            # whitespace), so raw "\n\n" in AR decodes is a surface-form mismatch that
            # feature-level metrics (MAUVE) punish as a fake distribution shift
            text = " ".join(tok.decode(row, skip_special_tokens=True).split())
            out.append({"id": p["id"], "text": text})
        print(f"AR/{args.decoding}: {len(out)}/{len(prefixes)}", flush=True)
    return out


def run_mdlm(args, prefixes):
    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    from chord.data.generators.flash_compat import install_if_missing

    install_if_missing()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    model = (
        AutoModelForMaskedLM.from_pretrained(
            args.checkpoint, trust_remote_code=True, torch_dtype=torch.float32
        )
        .to(args.device)
        .eval()
    )
    tok = AutoTokenizer.from_pretrained("gpt2")
    mask_index = int(model.config.vocab_size) - 1
    L = args.prefix_len + args.cont_len

    @torch.no_grad()
    def sample(pre_ids):  # pre_ids: LongTensor [B, L_pre]
        B = pre_ids.shape[0]

        def forward_logits(tokens):
            output = model(input_ids=tokens, timesteps=torch.zeros(B, device=args.device))
            return (
                output.logits
                if hasattr(output, "logits")
                else (output[0] if isinstance(output, tuple) else output)
            )

        eps = 1e-5
        x = torch.full((B, L), mask_index, dtype=torch.long, device=args.device)
        x[:, : args.prefix_len] = pre_ids  # clamp prefix (stays fixed in loop)
        timesteps = torch.linspace(1, eps, args.steps + 1, device=args.device)
        dt = (1 - eps) / args.steps
        cached = None
        for index in range(args.steps):
            t = timesteps[index]
            s = t - dt
            if cached is None:
                logits = forward_logits(x)
                logits[:, :, mask_index] = -1_000_000.0
                log_probs = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
                unmasked = x != mask_index
                log_probs[unmasked] = -1_000_000.0
                log_probs[unmasked, x[unmasked]] = 0.0
                cached = log_probs.exp()
            probabilities = cached * (t - s)
            probabilities[:, :, mask_index] = s
            proposed = torch.multinomial(
                probabilities.reshape(-1, probabilities.shape[-1]), 1
            ).reshape_as(x)
            next_x = torch.where(x != mask_index, x, proposed)
            if not torch.equal(next_x, x):
                cached = None
            x = next_x
        logits = forward_logits(x)
        logits[:, :, mask_index] = -1_000_000.0
        x = torch.where(x == mask_index, logits.argmax(-1), x)
        return x

    out = []
    for i in range(0, len(prefixes), args.batch_size):
        chunk = prefixes[i : i + args.batch_size]
        pre = torch.tensor(
            [p["prefix_ids"][: args.prefix_len] for p in chunk],
            dtype=torch.long,
            device=args.device,
        )
        ids = sample(pre)
        suffix = ids[:, args.prefix_len :]
        for p, row in zip(chunk, suffix):
            text = " ".join(tok.decode(row, skip_special_tokens=True).strip().split())
            out.append({"id": p["id"], "text": text})
        print(f"MDLM: {len(out)}/{len(prefixes)}", flush=True)
    return out


def _load_sedd(repo, model_path, device):
    import sys

    sys.path.insert(0, str(Path(repo).resolve()))
    import torch
    from load_model import load_model

    mp = Path(model_path)
    legacy = mp / "pytorch_model.bin"
    if mp.is_dir() and legacy.is_file() and not (mp / "model.safetensors").is_file():
        import graph_lib
        import noise_lib
        from model import SEDD
        from omegaconf import OmegaConf

        config = OmegaConf.load(mp / "config.json")
        model = SEDD(config).to(device)
        model.load_state_dict(torch.load(legacy, map_location=device, weights_only=True))
        graph = graph_lib.get_graph(config, device)
        noise = noise_lib.get_noise(config).to(device)
    else:
        model, graph, noise = load_model(model_path, device)
    return model.eval(), graph, noise


def run_sedd(args, prefixes):
    import torch
    from transformers import GPT2TokenizerFast

    from chord.data.generators.flash_compat import install_if_missing

    install_if_missing()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    model, graph, noise = _load_sedd(args.repo, args.model, args.device)
    import sampling

    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    L = args.prefix_len + args.cont_len
    input_locs = list(range(args.prefix_len))
    out = []
    for i in range(0, len(prefixes), args.batch_size):
        chunk = prefixes[i : i + args.batch_size]
        pre = torch.tensor(
            [p["prefix_ids"][: args.prefix_len] for p in chunk],
            dtype=torch.long,
            device=args.device,
        )

        def proj_fun(x):
            x[:, input_locs] = pre
            return x

        sampler = sampling.get_pc_sampler(
            graph,
            noise,
            (len(chunk), L),
            "analytic",
            args.steps,
            device=args.device,
            proj_fun=proj_fun,
        )
        ids = proj_fun(sampler(model))
        suffix = ids[:, args.prefix_len :]
        for p, row in zip(chunk, suffix):
            text = " ".join(tok.decode(row, skip_special_tokens=True).strip().split())
            out.append({"id": p["id"], "text": text})
        print(f"SEDD: {len(out)}/{len(prefixes)}", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["ar", "mdlm", "sedd"])
    ap.add_argument("--prefixes", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--decoding", default="nucleus")  # ar only
    ap.add_argument("--model", default="gpt2-medium")  # ar model OR sedd model dir
    ap.add_argument("--checkpoint")  # mdlm
    ap.add_argument("--repo")  # sedd official repo
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--prefix-len", type=int, default=128)
    ap.add_argument("--cont-len", type=int, default=128)
    ap.add_argument("--steps", type=int, default=128)  # dlm steps
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=20260621)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    prefixes = load_prefixes(args.prefixes)
    if args.mode == "ar":
        rows = run_ar(args, prefixes)
    elif args.mode == "mdlm":
        rows = run_mdlm(args, prefixes)
    else:
        rows = run_sedd(args, prefixes)
    rows = [r for r in rows if r["text"]]
    outp = Path(args.output)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(
        f"[continuation] mode={args.mode} decoding={args.decoding} "
        f"-> {len(rows)} continuations -> {outp}"
    )


if __name__ == "__main__":
    main()
