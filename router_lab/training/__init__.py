"""Finetuning optillm's router on labels this platform manufactured.

Depends on the results store, the label rule and the dataset registry; none of
them import this package. Its dependencies (torch, transformers, a tracker) are
an optional install group, so the harness and the dashboard keep working
without them:

    uv pip install --python .venv/bin/python3 -e '.[train]'
"""
