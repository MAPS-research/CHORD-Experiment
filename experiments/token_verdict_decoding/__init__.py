"""Token-space views of CHORD's readout (paper appendix "Hidden-State Signals
versus Token-Space Judgments", subsection "Token Counts and Next-Token
Predictions").

  decode        what the language-model head predicts at the exact position
                CHORD reads the hidden state from (next 1/3/5 tokens), for benign
                paraphrases versus sentence shuffling and contradiction; writes
                the per-string tallies behind fig_token_verdict_decoding

The token-count comparison (fig_hidden_vs_token_shuffle) reads the unigram-count
and CHORD rows of ``selectivity_by_family.csv``.

The token-count representation is ``backend: unigram-count`` in
``chord.embeddings``; its features come from
``experiments/token_verdict_decoding/configs/encoders_unigram_counts.yaml``.
"""
