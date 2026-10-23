"""Morphology, Not Frequency, Should Govern Vocabulary Allocation — experiment code.

Position paper for LP4FM @ NeurIPS 2026.  Three analyses:

* **Part A** — Indic fertility audit across mainstream, Indic-specialized and
  byte-level tokenizers (FLORES-200 devtest).
* **Part B** — morpheme–token boundary alignment against gold segmentations
  (the paper's original contribution).
* **Part C** — fixed-prompt-budget capability study on a 48 GB GPU.

Import side effects are deliberately absent; call ``morphtok.paths.setup_env()``
before importing transformers/datasets/torch so every cache stays in the repo.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
