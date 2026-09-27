"""Same-backbone LLM judges versus CHORD (paper appendix "Hidden-State Signals
versus Token-Space Judgments", tables tab:judge-counterfactual and
tab:judge-casestudy).

The frozen Qwen3.5-27B that CHORD reads hidden states from is served through
vLLM and asked for explicit ratings instead:

  judge_evaluation_set          direct 0-10 coherence rating of every passage
                                of the counterfactual evaluation set
  geval_evaluation_set          G-Eval (probability-weighted 1-5 coherence) on
                                the same passages
  judge_unconditional_generation  direct rating of the unconditional-generation
                                corpora, null-calibrated against the human reference

Per-document ratings go through ``experiments.counterfactual_eval.baseline_selectivity``
(scalar lane, |mean shift| statistic), i.e. the same null standardization and
benign-contrast detection rule as every row of Table 1.
"""
