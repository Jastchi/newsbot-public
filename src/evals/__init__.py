"""
Offline evaluations for the newsbot pipeline.

Scores story clustering against hand-labelled stories and summary
faithfulness (deterministic checks plus the LLM judge) against human
labels. Nothing here is imported by the pipeline itself.

Run from the repository root:
    PYTHONPATH=src uv run python -m evals --help
"""
