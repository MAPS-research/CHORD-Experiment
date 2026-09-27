# Table 2: ranking real generators (unconditional generation)

Every generator is sampled at 10 fresh seeds x 500 documents at its operating
point, and fold *k* of each generator is scored against fold *k* of a fresh
human reference, so every metric is reported as mean +- std over 10 disjoint
folds. Code and batch jobs live here; configs in
`experiments/generator_eval/configs/`; outputs in
`outputs/casestudy/unconditional_generation/`. Every command runs from the
repository root with `PYTHONPATH=$PWD`; the `.slurm` files are the batch jobs we
used (set `<SLURM_ACCOUNT>` and `CHORD_ROOT`).

## 1. Human reference (`build_human_pools.py`, CPU)

Streams OpenWebText parquet shards, drops every document that shares an 8-word
shingle with any text the students were trained, validated or evaluated on
(the `BAN_GLOBS` list in the script), and packs the survivors into 512-token
GPT-2 windows exactly as `experiments.generation_scoring.build_packed_reference` does.
Reference and held-out windows come from disjoint documents.

```bash
python -m experiments.generator_eval.build_human_pools --out outputs/casestudy/unconditional_generation/human
# -> human/reference_pool.jsonl  5,150 windows = 150 bandwidth-fitting ("dev") + 10 x 500 reference folds
# -> human/human_held.jsonl      5,000 windows = 10 x 500 held-out human folds
```

`human_pools.slurm` is the batch job.

## 2. Generator samples (`samples/`, GPU)

One job per generator, each writing
`outputs/casestudy/unconditional_generation/samples/r10-<generator>-s01..s10.jsonl`
(`r10` = the 10-fold protocol; seed `k` is `20260800 + k`). Finished seed files
are skipped, so a requeued job resumes.

| Job | Generator | Sampler | Operating point |
|---|---|---|---|
| `gen_gpt2medium.slurm`, `gen_gpt2large.slurm` | GPT-2 medium / large | `chord.data.generate_ar` | nucleus, T = 1.0, p = 0.95, 480-512 new tokens |
| `gen_mdlm.slurm` | MDLM (`kuleshov-group/mdlm-owt`) | `chord.data.generators.mdlm` | 512 steps, length 512 |
| `gen_sedd.slurm` | SEDD-small (`louaaron/sedd-small`) | `chord.data.generators.sedd` (official repo under `third_party/SEDD`) | 256 steps |
| `gen_langflow.slurm` | LangFlow-OWT | `chord.data.generators.langflow` (official repo under `third_party/LangFlow`) | 128 steps |
| `gen_elfL.slurm` | ELF-L-OWT | official ELF sampler, then `chord.data.generators.normalize_samples` | 64-step SDE, self-conditioning CFG 4 (`elf_l_sampling_steps64_cfg4.yaml`) |

`experiments/generator_eval/configs/runs.jsonl` ties every run to its file.

## 3. Features (GPU)

`featurize.yaml` lists every encoder read on the folds: the 27B CHORD encoder,
the distilled students, and the MAUVE (GPT-2) features of the baseline columns.

```bash
sbatch experiments/generator_eval/featurize_teacher.slurm     # 27B CHORD
sbatch experiments/generator_eval/featurize_students.slurm    # the distilled students
```

## 4. Fold scoring (CPU)

```bash
python -m experiments.generator_eval.score_chord_folds --config experiments/generator_eval/configs/featurize.yaml \
    --encoders qwen35-27b-prompteol-coherence-l62 chord-qwen3.5-2b-student      # CHORD columns (the first encoder sets the reference ranking)
python -m experiments.generator_eval.score_baselines_folds --config experiments/generator_eval/configs/featurize.yaml   # gen-PPL / MAUVE / unigram entropy
```

`score_chord.slurm` and `score_baselines.slurm` run the same (the former also
with the unbiased MMD estimator). Fold-level values:
`outputs/casestudy/unconditional_generation/scores/repeat_folds_{biased,unbiased,baselines}.csv`.
