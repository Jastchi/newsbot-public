# Evals

Offline evaluations for the pipeline. This package is not imported by
the pipeline; it only calls the existing agents.

Run from the repository root:

```bash
PYTHONPATH=src uv run python -m evals clustering
PYTHONPATH=src uv run python -m evals faithfulness
PYTHONPATH=src uv run python -m evals faithfulness --judge --yes
```

## What is scored

| Eval | Data | Metrics |
| --- | --- | --- |
| `clustering` | `data/clustering.json`: articles labelled with their story (`null` = no story) | ARI, pairwise precision / recall / F1 |
| `faithfulness` | `data/faithfulness.json`: summaries labelled faithful / unfaithful with the injected error type; sources come from the story's articles in `clustering.json` | precision, recall, F1, specificity, Cohen's kappa, recall per error type |

Clustering runs the production agent (embedding model, geo penalty,
algorithm) with the production config defaults. Override with
`--algorithm`, `--similarity-threshold` and `--embedding-model` to compare settings.
Unclustered (noise) articles count as singletons.

Faithfulness scores these detectors against the human labels:

- `numbers_dates`: figures, weekdays and months in the summary that the
  source never states.
- `names`: the pipeline's `NameValidationAgent` (spaCy NER).
- `deterministic`: either of the two above.
- `judge` (`--judge`): the pipeline's `JudgeAgent` with source checking.
  It makes one billed API call per case, so it also needs `--yes`.
  Set the provider and model with `--provider` and `--judge-model`.

## Regression runs

`--save PATH` writes the metrics. `--baseline PATH` exits with 1 when a
baseline metric dropped by more than `--tolerance` (default 0.05) or is
missing from this run, and warns when the clustering settings differ.
Both options go after the eval name. Baselines are in `baselines/`.
Re-run after every prompt, model or clustering-setting change:

```bash
PYTHONPATH=src uv run python -m evals clustering --baseline src/evals/baselines/clustering.json
PYTHONPATH=src uv run python -m evals faithfulness --baseline src/evals/baselines/faithfulness.json
```

The faithfulness baseline includes the `judge.*` metrics. The free run,
without `--judge`, checks everything else and reports that the
`judge.*` metrics were left out. To check those too, run with
`--judge --yes` (billed). A judge run that fails open on any call is not
scored.

The judge is an LLM, so its scores move between runs, and on a set this
small one flipped verdict can move kappa by more than the default
tolerance. A judge regression run can pass `--tolerance 0.1`, which also
loosens the other metrics in that run.

## Per-config runs

Each config in the database has its own clustering and LLM settings,
language and country. Run an eval with one config's settings:

```bash
PYTHONPATH=src uv run python -m evals clustering --config <key>
PYTHONPATH=src uv run python -m evals faithfulness --config <key> --save src/evals/baselines/faithfulness.<key>.json
```

- `--config KEY` reads the config from the database (`ENVIRONMENT=dev`
  selects the dev schema). `--config-file PATH` reads a JSON dump of a
  `ConfigModel` instead, which needs no database.
- Clustering uses the config's `story_clustering` settings; a flag such
  as `--similarity-threshold` still overrides them. Faithfulness uses
  the config's `llm` settings: the spaCy model for the name check, and
  for `--judge` the provider and judge model (`--provider` and
  `--judge-model` override them).
- The seed data is English. A config in another language is refused
  until you give it labelled data of its own with `--data DIR`, a
  directory holding `clustering.json` (with `"language": "<code>"`) and
  `faithfulness.json`.
- Keep one baseline per config, for example
  `baselines/clustering.<key>.json`. A comparison warns when the saved
  config key or settings differ from the run.

## Layout

| File | Role |
| --- | --- |
| `__main__.py` | Command line: options, `--yes` gate, exit codes |
| `clustering.py`, `faithfulness.py` | The two evaluations; each returns a `Result` |
| `metrics.py` | Pure scoring: ARI, pairwise scores, detector scores, kappa, number/date grounding, `regressions()` |
| `configs.py` | `Subject`: the config and data directory a run uses, and how they are loaded |
| `datasets.py` | Pydantic models that validate `data/` when it is loaded |
| `results.py` | `Result`, table output, comparison with a baseline |

Exit codes: 0 ok, 1 regression, 2 the run could not be scored (invalid
data, missing `--yes`, a judge that failed open, a baseline with nothing left to compare).

Tests are in `tests/evals_tests/`. They include checks that fail if the
pipeline renames a method the evals call, or rewords the judge prompt the
faithfulness eval copies.

## Data

The files in `data/` are an authored seed set (fictional events, six
stories, 29 summaries), not scraped news and not independently labelled.
Replace or extend them with real articles and summaries you have labelled
yourself before treating the numbers as a measure of production quality.
The seed set is small: one case moves a faithfulness metric by about
0.03.
