"""Modern embedding and evaluator baselines on the counterfactual evaluation set
(paper appendix "Modern Embedding and Evaluator Baselines", table tab:modern-baselines).

  Instruction-tuned embeddings (Qwen3-Embedding-8B, e5-mistral-7b-instruct,
  gte-Qwen2-7B-instruct), each with no instruction, a similarity instruction and
  a coherence instruction: final-layer last-token embeddings, L2-normalized,
  scored with CHORD's own RBF-MMD endpoint (configs in
  ``experiments/embedding_evaluator_baselines/configs/encoders_*.yaml``).

  Source-free scalar evaluators, scored with the |mean shift| lane of
  ``experiments.counterfactual_eval.baseline_selectivity``:
    unieval_scores    UniEval fluency (unieval-sum) and zero-shot coherence
                      (unieval-intermediate), and their average
    bartscore_scores  hypothesis-only BARTScore under bart-large-cnn
"""
