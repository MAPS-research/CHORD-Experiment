"""Source-conditioned QA faithfulness: building the evaluation set scored in the appendix.

Steps: ``build_verbatim_conditions`` (parents + verbatim edits, CPU) ->
``paraphrase_benign_control`` (LLM, draw B) -> ``build_paraphrase_conditions``
(LLM draw A + wrong-fact edits) -> ``assemble_evalset`` (layout for scoring).
Shared settings: ``experiments/qa_faithfulness/configs/build.yaml``.
"""
