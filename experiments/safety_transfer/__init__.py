"""Transfer of CHORD's corpus-level recipe to unsafe-content prevalence.

Appendix "Preliminary Transfer to Unsafe-Content Prevalence": the coherence
prompt is replaced by a safety prompt and the same RBF-MMD endpoint asks whether
a corpus of safe assistant responses changes when a small fraction is replaced
by unsafe ones. Pipeline, in order:

    download_sources           PKU-SafeRLHF release files
    build_corpus               prevalence ladder with matched controls
    featurize pool / compose   one forward pass per unique response
    score_safety_classifiers   toxic-bert and Granite Guardian per-response scores
    paired_contrast            CHORD rows (unsafe vs matched safe, paired draws)
    mauve_scores               MAUVE on GPT-2-large features

The baseline rows (FBD, FI-KL, classifiers) are scored with
``experiments.counterfactual_eval.baseline_selectivity`` and
``experiments/safety_transfer/configs/baseline_selectivity.yaml``.
"""
