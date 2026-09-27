"""Join the per-readout outputs of ``paired_contrast`` into the files ``tables`` reads.

When each readout runs as its own job (``--out-suffix _<encoder>``), this
concatenates ``safety_{arms,contrast,calibration}_<encoder>.csv`` into the
unsuffixed CSVs, in the order below, and stops if a part is missing.

    python -m experiments.safety_transfer.merge_contrast_parts \
        --corpus-dir outputs/experiments/safety_transfer
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

ENCODERS = [
    "q35-27b-safety-w1",
    "q35-27b-safety-w2",
    "q35-27b-safety-w3",
    "q35-27b-coherence",
    "q35-27b-meaning",
]
STEMS = ["safety_arms", "safety_contrast", "safety_calibration"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", required=True)
    parser.add_argument("--encoders", default=",".join(ENCODERS))
    args = parser.parse_args()
    sel = Path(args.corpus_dir) / "selectivity"
    encoders = [e.strip() for e in args.encoders.split(",") if e.strip()]

    for stem in STEMS:
        rows, header = [], None
        for enc in encoders:
            part = sel / f"{stem}_{enc}.csv"
            if not part.exists():
                raise SystemExit(f"missing part: {part}")
            with part.open() as fh:
                reader = csv.DictReader(fh)
                if header is None:
                    header = reader.fieldnames
                elif reader.fieldnames != header:
                    raise SystemExit(f"header mismatch in {part}")
                rows.extend(reader)
        out = sel / f"{stem}.csv"
        with out.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=header)
            writer.writeheader()
            writer.writerows(rows)
        print(f"[merge] {out}: {len(rows)} rows from {len(encoders)} readouts", flush=True)


if __name__ == "__main__":
    main()
