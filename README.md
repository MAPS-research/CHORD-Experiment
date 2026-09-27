# CHORD experiments

This repository reproduces the tables and figures of the CHORD paper. CHORD
embeds each passage with a frozen language model under a coherence-oriented
PromptEOL readout and compares a generated corpus with a human reference corpus
using RBF-MMD. The paper tests whether this distance detects relation- and
discourse-level coherence failures that likelihood statistics and standard
distributional metrics miss, while staying stable under benign paraphrasing.

The evaluation has four parts: a counterfactual meta-evaluation that applies
matched harmful and benign edits to the same human passages (Table 1,
Figures 3–4); real generators ranked on unconditional generation (Table 2) and
prefix continuation (Figure 5); agreement with human judgments (Table 3); and
the appendix studies.

The metric itself, its distilled students and the distillation pipeline live in
the library repository, [CHORD](https://github.com/NYUSH-ML/CHORD)
(`pip install chord-metric`); this repository only adds the experiments.

## Installation

```bash
git clone git@github.com:NYUSH-ML/CHORD-Experiment.git && cd CHORD-Experiment
pip install -e .     # also installs the library, chord-metric[hf,distill], from PyPI
export PYTHONPATH=$PWD HF_HOME=/path/to/cache/huggingface
```

The `.slurm` jobs activate the conda environments we used: `chord-gpu`, and
`chord-generators` for the three diffusion-generator samplers. Both are created
from the library repository (`environment-gpu.yml`,
`environment-generators.yml`), which is also needed to train a student
(`distillation/`):

```bash
cd .. && git clone git@github.com:NYUSH-ML/CHORD.git
cd CHORD && conda env create -f environment-gpu.yml && conda activate chord-gpu
cd ../CHORD-Experiment && pip install -e .
```

Every command below runs from this repository's root. Texts, features and
results are read from and written to `data/` and `outputs/` here (gitignored).

## Data

The corpora behind every table ship as a Hugging Face dataset, [`mikezhu/chord-experiments-data`](https://huggingface.co/datasets/mikezhu/chord-experiments-data), whose tree mirrors
`outputs/`, so the files land where the configs expect them:

```bash
python scripts/download_data.py                            # all texts (mikezhu/chord-experiments-data)
python scripts/download_data.py --features                 # + cached 27B features of Table 2's folds, gen-PPL cache
python scripts/download_data.py --checkpoint qwen3.5-2b    # + a trained student (qwen3.5-2b or qwen3.5-0.8b)
```

`MANIFEST-experiments.tsv` in the bundle lists every file with its sha256, and
`download_data.py` checks it after download. To score with a student you
trained yourself, point the student configs (`experiments/student_eval/configs/`,
`experiments/generator_eval/configs/featurize.yaml`) at its `final/` directory,
or link the library's `outputs/distill` into this repository's `outputs/`.

## A first check

With the released features, Table 2's CHORD-27B column is re-scored on a CPU,
without loading the 27B model:

```bash
python scripts/download_data.py --features
python -m experiments.generator_eval.score_chord_folds --config experiments/generator_eval/configs/featurize.yaml \
    --encoders qwen35-27b-prompteol-coherence-l62
```

Fold-level values are written to
`outputs/casestudy/unconditional_generation/scores/repeat_folds_biased.csv`.

## Reproducing the paper

| Result | Package | Output |
|---|---|---|
| Table 1: counterfactual robustness | `counterfactual_eval/`, `backbone_families/` | `outputs/counterfactual/meta_eval/selectivity/` |
| Figure 3: representation × distance, power versus corpus size | `representation_distance/` | `outputs/experiments/representation_distance/` |
| Figure 4: backbone size, extraction method | `counterfactual_eval/`, `readout_ablations/` | `outputs/counterfactual/meta_eval/selectivity/` |
| Table 2: unconditional generation | `generator_eval/` | `outputs/casestudy/unconditional_generation/scores/` |
| Figure 5: prefix continuation | `prefix_continuation/` | `outputs/casestudy/prefix_continuation/` |
| Table 3: agreement with human judgments | `human_agreement/` | `outputs/casestudy/human_agreement/` |
| a distilled student's Table 1 row and Table 2 column | `student_eval/` | merged into the Table 1 and Table 2 outputs |
| appendix studies | one package each (see [Appendix experiments](#appendix-experiments)) | `outputs/experiments/<name>/` |

Notes:

- Featurizing and sampling need a GPU; the 27B encoder needs one with ≥60 GB
  memory. Scoring reads cached features and runs on CPU.
- Steps that sample from an LLM editor or a generator are stochastic, so
  re-derived numbers match the paper up to sampling noise.
- The `.slurm` files under `experiments/` are the batch jobs we used; set
  `<SLURM_ACCOUNT>`, `<GPU_PARTITION>`, `<CPU_PARTITION>` and `CHORD_ROOT` for
  your cluster, or run their Python entry points directly.
- Encoder ids spell out the readout: `qwen35-27b-prompteol-coherence-l62` is
  Qwen3.5-27B read at hidden layer 62 of 64 (−3) after the `coherence`
  PromptEOL template; [`chord-qwen3.5-2b-student`](https://huggingface.co/mikezhu/chord-qwen3.5-2b-student) and
  [`chord-qwen3.5-0.8b-student`](https://huggingface.co/mikezhu/chord-qwen3.5-0.8b-student) are the distilled students.

### Shared inputs

```bash
python scripts/download_openwebtext.py            # OpenWebText documents
python -m experiments.generation_scoring.build_diverse_corpus    # 3-domain (OWT / Wikipedia / Reddit) human pool
```

LLM-edited perturbations use Qwen3-30B-A3B behind an OpenAI-compatible endpoint
(e.g. `vllm serve Qwen/Qwen3-30B-A3B --served-model-name qwen3-30b-editor`);
the endpoint and model name live in `experiments/counterfactual_eval/configs/perturbations_qwen_editor.yaml`.

### Table 1: counterfactual robustness

```bash
python -m chord.data.split_passages --config experiments/counterfactual_eval/configs/select_source_passages.yaml
python -m chord.data.generate_perturbations   --config experiments/counterfactual_eval/configs/perturbations_qwen_editor.yaml
python -m experiments.counterfactual_eval.validate_perturbations   --config experiments/counterfactual_eval/configs/validate_perturbations.yaml
python -m chord.data.counterfactual   --config experiments/counterfactual_eval/configs/build_evalset.yaml
python -m experiments.counterfactual_eval.inspect_corpus --corpus-dir outputs/counterfactual/meta_eval --samples 5

# CHORD rows: 27B, 9B, 2B student (GPU)
for c in encoders_chord_27b encoders_chord_9b encoders_chord_qwen3.5-2b; do
  python -m experiments.counterfactual_eval.featurize --config experiments/counterfactual_eval/configs/$c.yaml --corpus-dir outputs/counterfactual/meta_eval
done
python -m experiments.counterfactual_eval.selectivity   --config experiments/counterfactual_eval/configs/selectivity.yaml --corpus-dir outputs/counterfactual/meta_eval
python -m experiments.counterfactual_eval.dose_response --corpus-dir outputs/counterfactual/meta_eval

# baseline rows: gen-PPL, unigram entropy, MAUVE (GPT-2 / ELECTRA), FBD (BERT), MMD (MiniLM)
python -m experiments.counterfactual_eval.featurize --config experiments/counterfactual_eval/configs/encoders_baselines.yaml --corpus-dir outputs/counterfactual/meta_eval
python -m experiments.counterfactual_eval.score_genppl --config experiments/counterfactual_eval/configs/genppl.yaml --score-config experiments/counterfactual_eval/configs/selectivity.yaml --corpus-dir outputs/counterfactual/meta_eval
python -m experiments.counterfactual_eval.baseline_selectivity --config experiments/counterfactual_eval/configs/baseline_selectivity.yaml --corpus-dir outputs/counterfactual/meta_eval
```

Results land in `outputs/counterfactual/meta_eval/selectivity/`
(`selectivity_by_family.csv` holds the CHORD rows; Table 1 reports the
highest-severity condition, `dose3`, per family).
`experiments/counterfactual_eval/run_meta_eval.sh` chains the CHORD-27B path end
to end.

Other backbone families in Table 1 (Gemma-2, Mistral, Llama-3.1):

```bash
for c in gemma2 mistral llama31; do
  python -m experiments.counterfactual_eval.featurize --config experiments/backbone_families/configs/encoders_$c.yaml --corpus-dir outputs/counterfactual/meta_eval
done
python -m experiments.counterfactual_eval.selectivity --config experiments/counterfactual_eval/configs/selectivity.yaml --corpus-dir outputs/counterfactual/meta_eval \
    --only-encoders gemma2-9b-prompteol-coherence-l40,gemma2-27b-prompteol-coherence-l44,mistral-8b-prompteol-coherence-l34,mistral-24b-prompteol-coherence-l38,llama31-8b-prompteol-coherence-l30
```

### Figure 3: representation × distance, and power versus corpus size

The factorial crosses four representations (GPT-2-large, ELECTRA, BERT, CHORD)
with four distances (RBF-MMD, energy, Fréchet, k-means KL) on the
counterfactual set; the power study subsamples the corpus and records detection
per size. Both read the features written by the Table 1 commands.
`--distances rbf_mmd,energy` restricts either script to a subset of distances.

```bash
python -m experiments.representation_distance.factorial --corpus-dir outputs/counterfactual/meta_eval --out-dir outputs/experiments/representation_distance
python -m experiments.representation_distance.power_vs_corpus_size --rep qwen35-27b-prompteol-coherence-l62 --corpus-dir outputs/counterfactual/meta_eval --out-dir outputs/experiments/representation_distance
```

### Figure 4: backbone size and extraction method

The Qwen3.5 backbone ladder (panel a) and the prompt-factorization analysis of
the appendix reuse the Table 1 evaluation set:

```bash
python -m experiments.counterfactual_eval.featurize   --config experiments/counterfactual_eval/configs/encoders_backbone_size.yaml --corpus-dir outputs/counterfactual/meta_eval
python -m experiments.counterfactual_eval.selectivity --config experiments/counterfactual_eval/configs/selectivity_backbone_size.yaml --corpus-dir outputs/counterfactual/meta_eval --out-suffix _backbone_size
python -m experiments.counterfactual_eval.featurize   --config experiments/counterfactual_eval/configs/encoders_prompt_factorization.yaml --corpus-dir outputs/counterfactual/meta_eval
python -m experiments.counterfactual_eval.prompt_split_eval --corpus-dir outputs/counterfactual/meta_eval --draws 80 --out-suffix _prompt_factorization
python -m experiments.counterfactual_eval.analyze_prompt_factorization --config experiments/counterfactual_eval/configs/encoders_prompt_factorization.yaml \
    --split-csv outputs/counterfactual/meta_eval/selectivity/prompt_split_eval_prompt_factorization.csv --out-dir outputs/counterfactual/meta_eval/selectivity/prompt_factorization
```

Extraction methods (panel b, and the GPT-2 extraction grid of the appendix).
The Qwen3.5-27B rows also need `experiments/counterfactual_eval/configs/encoders_prompt_wording.yaml`
and `experiments/counterfactual_eval/configs/encoders_no_prompt.yaml` featurized
and scored.

```bash
python -m experiments.counterfactual_eval.featurize --config experiments/readout_ablations/configs/encoders_gpt2_extraction.yaml --corpus-dir outputs/counterfactual/meta_eval
python -m experiments.counterfactual_eval.selectivity --config experiments/counterfactual_eval/configs/selectivity.yaml --corpus-dir outputs/counterfactual/meta_eval --out-suffix _gpt2_extraction
```

### Table 2: unconditional generation

Every generator is sampled at 10 fresh seeds × 500 documents at its operating
point and scored fold by fold against a fresh human reference (150
bandwidth-fitting windows plus 10 × 500 packed-512 OWT windows, overlap-filtered
against all training and evaluation text). Everything is in
`experiments/generator_eval/` (see its README); the short form:

```bash
python -m experiments.generator_eval.build_human_pools --out outputs/casestudy/unconditional_generation/human
experiments/generator_eval/samples/gen_{gpt2medium,gpt2large,elfL,mdlm,sedd,langflow}.slurm      # samples
experiments/generator_eval/featurize_teacher.slurm ; experiments/generator_eval/featurize_students.slurm   # 27B and student features
python -m experiments.generator_eval.score_chord_folds --config experiments/generator_eval/configs/featurize.yaml --encoders qwen35-27b-prompteol-coherence-l62 chord-qwen3.5-2b-student
python -m experiments.generator_eval.score_baselines_folds --config experiments/generator_eval/configs/featurize.yaml   # gen-PPL / MAUVE / entropy on the same folds
```

Fold-level values: `outputs/casestudy/unconditional_generation/scores/repeat_folds_{biased,unbiased,baselines}.csv`.

### Figure 5: prefix continuation

```bash
python -m experiments.prefix_continuation.build_continuation_corpus
python -m experiments.prefix_continuation.generate_continuation --mode ar --decoding nucleus --model gpt2-medium \
    --prefixes outputs/casestudy/prefix_continuation/prefixes.jsonl --prefix-len 128 --cont-len 128
python -m experiments.prefix_continuation.generate_continuation --mode sedd --steps 128 --repo <SEDD clone> --model <sedd-small snapshot> ...
python -m experiments.prefix_continuation.generate_continuation --mode mdlm --steps 128 --checkpoint kuleshov-group/mdlm-owt ...
python -m chord.featurize --config experiments/prefix_continuation/configs/prefix_continuation_chord_27b.yaml --only-encoder qwen35-27b-prompteol-coherence-l62
python -m experiments.generation_scoring.score_chord --config experiments/prefix_continuation/configs/prefix_continuation_chord_27b.yaml --encoders qwen35-27b-prompteol-coherence-l62
python -m chord.featurize --config experiments/prefix_continuation/configs/prefix_continuation_all_metrics.yaml
python -m experiments.generation_scoring.score --config experiments/prefix_continuation/configs/prefix_continuation_all_metrics.yaml
python -m experiments.generation_scoring.score_generation --config experiments/prefix_continuation/configs/prefix_continuation_all_metrics.yaml
```

### Table 3: agreement with human judgments

Uses the human study released with MAUVE (Pillutla et al., 2021): the anonymized
judgment CSV linked from `human_evaluation.md` in `krishnap25/mauve-experiments`,
and GPT-2's `webtext.test.jsonl` for the reference pool. No new human annotation
is collected.

```bash
python -m experiments.human_agreement.build_mauve_human_eval --csv <mauve human_eval CSV> --webtext <webtext.test.jsonl> --output-dir outputs/casestudy/human_agreement
python -m chord.featurize        --config experiments/human_agreement/configs/human_agreement.yaml
python -m experiments.generation_scoring.score_generation --config experiments/human_agreement/configs/human_agreement.yaml
python -m experiments.generation_scoring.score_chord --config experiments/human_agreement/configs/human_agreement.yaml --encoders qwen35-27b-prompteol-coherence-l62,chord-qwen3.5-2b-student,minilm-cls
python -m experiments.generation_scoring.score            --config experiments/human_agreement/configs/human_agreement.yaml
python -m experiments.human_agreement.analyze_human_agreement --dir outputs/casestudy/human_agreement
```

### A distilled student's Table 1 row and Table 2 column

Train the students with the library (`distillation/` in CHORD), or download
them with `python scripts/download_data.py --checkpoint qwen3.5-2b`, then:

```bash
STUDENT=qwen3.5-2b sbatch experiments/student_eval/eval.slurm      # or STUDENT=qwen3.5-0.8b
```

### Appendix experiments

Each study below has its own package under `experiments/<name>/` and configs
under `experiments/<name>/configs/`; outputs go to `outputs/experiments/<name>/`.

#### Compute and runtime cost

```bash
python -m experiments.compute_cost.bench_cost --docs 500 --out outputs/casestudy/unconditional_generation/scores/cost.csv   # one GPU
```

#### Source-conditioned faithfulness extension

Needs SQuAD v1.1 dev at `data/squad/dev-v1.1.json`; the two paraphrase steps
call the Qwen3-30B-A3B editor. The built evaluation set is also in the data
bundle.

```bash
python -m experiments.qa_faithfulness.build_verbatim_conditions  --config experiments/qa_faithfulness/configs/build.yaml
python -m experiments.qa_faithfulness.paraphrase_benign_control   --config experiments/qa_faithfulness/configs/build.yaml   # LLM editor
python -m experiments.qa_faithfulness.build_paraphrase_conditions --config experiments/qa_faithfulness/configs/build.yaml   # LLM editor
python -m experiments.qa_faithfulness.assemble_evalset            --config experiments/qa_faithfulness/configs/build.yaml
experiments/qa_faithfulness/qa_faithfulness.slurm ; experiments/qa_faithfulness/qa_baselines.slurm   # featurize + score all encoders
```

#### Effect of extraction layer

On the Table 1 evaluation set, with each backbone's Table 1 encoder config and
scoring endpoint; only the read depth varies. Step 1 needs a GPU
(`featurize.slurm`), step 2 is CPU-only (`score.slurm`).

```bash
python -m experiments.layer_depth.featurize --encoder-config experiments/counterfactual_eval/configs/encoders_chord_27b.yaml \
    --corpus-dir outputs/counterfactual/meta_eval --out-dir outputs/experiments/layer_depth/qwen35_27b
python -m experiments.layer_depth.selectivity_by_layer --sweep-dir outputs/experiments/layer_depth/qwen35_27b \
    --rng-encoder qwen35-27b-prompteol-coherence-l62
# 9B: encoders_chord_9b.yaml, --out-dir/--sweep-dir .../qwen35_9b, --rng-encoder qwen35-9b-prompteol-coherence-l30
```

#### Effect of coherence-failure position on detection

One topic-drift sentence is inserted at the prefix, middle or suffix of each
passage and five Qwen3.5-9B readouts are compared. The built corpus is in the
data bundle; rebuilding it from the current evaluation set gives a corpus with
the same design but different passages.

```bash
python -m experiments.position_robustness.build_corpus --source-corpus outputs/counterfactual/meta_eval --perturbations data/interim/perturbations_qwen_editor.jsonl
python -m experiments.counterfactual_eval.featurize --config experiments/position_robustness/configs/encoders.yaml --corpus-dir outputs/experiments/position_robustness
python -m experiments.position_robustness.analyze --corpus-dir outputs/experiments/position_robustness
```

#### Sensitivity to evaluation window length

```bash
# Table 2's folds (experiments/generator_eval) cut at sentence boundaries to each length
python -m experiments.window_length.build_truncated_corpora --lengths 128,256,384,512,1024
for L in 128 256 384 512 1024; do D=outputs/experiments/window_length_folds/L$L
  python -m chord.featurize --config $D/chord.yaml      # GPU: window_length_featurize.slurm
  python -m chord.featurize --config $D/mauve.yaml
  python -m experiments.window_length.score_folds --dir $D   # CPU: window_length_score.slurm
done
python -m experiments.window_length.collect
```

#### Hidden-state signals versus token-space judgments

The judge and G-Eval steps call Qwen3.5-27B through an OpenAI-compatible server.

```bash
vllm serve Qwen/Qwen3.5-27B --served-model-name qwen35-27b-judge --max-model-len 4096
python -m experiments.llm_judge.judge_evaluation_set --config experiments/llm_judge/configs/judge.yaml --corpus-dir outputs/counterfactual/meta_eval
python -m experiments.llm_judge.geval_evaluation_set --config experiments/llm_judge/configs/geval.yaml --corpus-dir outputs/counterfactual/meta_eval
python -m experiments.counterfactual_eval.baseline_selectivity --config experiments/llm_judge/configs/baseline_selectivity_llm_judges.yaml --corpus-dir outputs/counterfactual/meta_eval --out-suffix _llm_judges
python -m experiments.llm_judge.judge_unconditional_generation --config experiments/llm_judge/configs/judge_unconditional_generation.yaml
# bag-of-words control and next-token verdicts at CHORD's readout position
python -m experiments.counterfactual_eval.featurize --config experiments/token_verdict_decoding/configs/encoders_unigram_counts.yaml --corpus-dir outputs/counterfactual/meta_eval
python -m experiments.counterfactual_eval.selectivity --config experiments/counterfactual_eval/configs/selectivity.yaml --corpus-dir outputs/counterfactual/meta_eval --only-encoders unigram-count
python -m experiments.token_verdict_decoding.decode --analysis content --model Qwen/Qwen3.5-27B --revision fc05daec18b0a78c049392ed2e771dde82bdf654 --corpus-dir outputs/counterfactual/meta_eval
```

#### Modern embedding and evaluator baselines

```bash
for c in qwen3_embedding_8b e5_mistral_7b gte_qwen2_7b; do
  python -m experiments.counterfactual_eval.featurize --config experiments/embedding_evaluator_baselines/configs/encoders_$c.yaml --corpus-dir outputs/counterfactual/meta_eval
done
for ENC in qwen3emb-8b-{none,sts,coh} e5mistral-7b-{none,sts,coh} gteqwen2-7b-{none,sts,coh}; do
  python -m experiments.counterfactual_eval.selectivity --config experiments/counterfactual_eval/configs/selectivity.yaml --corpus-dir outputs/counterfactual/meta_eval --only-encoders $ENC --out-suffix _modern_$ENC
done
python -m experiments.embedding_evaluator_baselines.unieval_scores --config experiments/embedding_evaluator_baselines/configs/unieval.yaml --corpus-dir outputs/counterfactual/meta_eval
python -m experiments.embedding_evaluator_baselines.bartscore_scores --config experiments/embedding_evaluator_baselines/configs/bartscore.yaml --corpus-dir outputs/counterfactual/meta_eval
python -m experiments.counterfactual_eval.baseline_selectivity --config experiments/embedding_evaluator_baselines/configs/baseline_selectivity_evaluators.yaml --corpus-dir outputs/counterfactual/meta_eval --out-suffix _evaluators
```

#### Transfer to unsafe-content prevalence

Uses PKU-SafeRLHF (CC-BY-NC-4.0), fetched by the first command.

```bash
python -m experiments.safety_transfer.download_sources
python -m experiments.safety_transfer.build_corpus --config experiments/safety_transfer/configs/corpus.yaml
python -m experiments.safety_transfer.featurize pool --config experiments/safety_transfer/configs/encoders_chord_prompts.yaml --corpus-dir outputs/experiments/safety_transfer
python -m experiments.safety_transfer.featurize pool --config experiments/safety_transfer/configs/encoders_gpt2_large.yaml --corpus-dir outputs/experiments/safety_transfer
python -m experiments.safety_transfer.featurize compose --corpus-dir outputs/experiments/safety_transfer
python -m experiments.safety_transfer.score_safety_classifiers --corpus-dir outputs/experiments/safety_transfer --scorer toxbert
python -m experiments.safety_transfer.score_safety_classifiers --corpus-dir outputs/experiments/safety_transfer --scorer guardian --batch-size 16
python -m experiments.safety_transfer.paired_contrast --config experiments/safety_transfer/configs/paired_contrast.yaml --corpus-dir outputs/experiments/safety_transfer
python -m experiments.counterfactual_eval.baseline_selectivity --config experiments/safety_transfer/configs/baseline_selectivity.yaml --corpus-dir outputs/experiments/safety_transfer
python -m experiments.safety_transfer.mauve_scores --config experiments/safety_transfer/configs/mauve.yaml --corpus-dir outputs/experiments/safety_transfer
```

## Repository layout

| Path | Purpose |
|---|---|
| `experiments/counterfactual_eval/` | Table 1: build the counterfactual evaluation set, featurize, null calibration, selectivity, dose–response; backbone-size and prompt ablations |
| `experiments/backbone_families/` | Table 1's Gemma-2, Mistral and Llama-3.1 encoder configs |
| `experiments/representation_distance/` | Figure 3: representation × distance factorial, power versus corpus size |
| `experiments/readout_ablations/` | Figure 4b and the appendix GPT-2 extraction grid |
| `experiments/generator_eval/` | Table 2: ranking real generators over 10 folds |
| `experiments/generation_scoring/` | scoring shared by Table 2, Figure 5 and Table 3 (CHORD and baselines on generation runs) |
| `experiments/prefix_continuation/` | Figure 5 |
| `experiments/human_agreement/` | Table 3 (MAUVE human study) |
| `experiments/student_eval/` | a trained student's Table 1 row, Table 2 column and comparison with the teacher |
| `experiments/single_fold/` | the earlier single-fold generation protocol (quick student check) |
| `experiments/metric_utils/` | the gen-PPL scorer and interval statistics shared by several experiments |
| `experiments/<appendix name>/` | one package per appendix study, each with its own `configs/` |
| `scripts/` | `download_data.py` (released texts, features, students), `download_openwebtext.py` |
