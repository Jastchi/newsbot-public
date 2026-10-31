"""
Run the evaluations.

    PYTHONPATH=src uv run python -m evals clustering
    PYTHONPATH=src uv run python -m evals faithfulness
    PYTHONPATH=src uv run python -m evals faithfulness --judge --yes
    PYTHONPATH=src uv run python -m evals clustering --config <key>

Each eval takes --save PATH to write the metrics and --baseline PATH to
fail (exit code 1) when a metric dropped, or went missing, against a
saved run. --config KEY (from the database) or --config-file PATH (a
JSON dump of a ConfigModel) runs with that config's settings, and --data
DIR with labelled data of its own. The judge run calls an LLM API once
per case, which may be billed, so it needs --judge and --yes.
"""

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

from evals import clustering, configs, faithfulness
from evals.metrics import DEFAULT_TOLERANCE
from evals.results import EvalError, Result, compare
from newsbot.llm_provider.gemini import DEFAULT_GEMINI_MODEL

EXIT_REGRESSION = 1
EXIT_UNSCORED = 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="evals", description=__doc__)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--save", type=Path, help="write results here")
    common.add_argument("--baseline", type=Path, help="compare to this run")
    common.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    which = common.add_mutually_exclusive_group()
    which.add_argument("--config", help="config key, read from the database")
    which.add_argument(
        "--config-file",
        type=Path,
        help="JSON dump of a ConfigModel",
    )
    common.add_argument("--data", type=Path, help="labelled data directory")
    sub = parser.add_subparsers(dest="eval", required=True)

    cl = sub.add_parser(
        "clustering",
        parents=[common],
        help="ARI vs hand-labelled stories",
    )
    cl.add_argument("--algorithm", choices=["greedy", "dbscan", "hdbscan"])
    cl.add_argument("--similarity-threshold", type=float)
    cl.add_argument("--embedding-model")

    fa = sub.add_parser(
        "faithfulness",
        parents=[common],
        help="summary faithfulness",
    )
    fa.add_argument("--judge", action="store_true", help="run the LLM judge")
    fa.add_argument("--yes", action="store_true", help="accept API cost")
    # unset: the config's own, or Gemini without a config
    fa.add_argument("--provider")
    fa.add_argument("--judge-model")
    return parser


def _judge_options(
    args: argparse.Namespace,
    subject: configs.Subject,
) -> faithfulness.JudgeOptions:
    if subject.key:
        llm = subject.config.llm
        return faithfulness.JudgeOptions(
            args.provider or llm.provider,
            args.judge_model or llm.judge_model,
        )
    return faithfulness.JudgeOptions(
        args.provider or "gemini",
        args.judge_model or DEFAULT_GEMINI_MODEL,
    )


def _run(args: argparse.Namespace) -> Result:
    subject = configs.from_args(args)
    if args.eval == "clustering":
        return clustering.run(
            {
                option: value
                for option in clustering.OPTIONS
                if (value := getattr(args, option)) is not None
            },
            subject,
        )
    if not args.judge:
        return faithfulness.run(subject=subject)
    judge = _judge_options(args, subject)
    if not args.yes:
        count = len(subject.load_faithfulness()[1].cases)
        msg = (
            f"--judge makes {count} {judge.provider} calls "
            f"({judge.model or 'the config model'}), which may be billed; "
            "re-run with --yes to proceed"
        )
        raise EvalError(msg)
    return faithfulness.run(judge, subject)


def main(argv: list[str] | None = None) -> int:
    """
    Run one evaluation and apply the baseline.

    Args:
        argv: Command line; defaults to sys.argv.

    Returns:
        0 on success, 1 on a regression, 2 if the run could not be
        scored.

    """
    args = _parser().parse_args(argv)
    load_dotenv()
    try:
        result = _run(args)
    except EvalError as err:
        sys.stderr.write(f"{err}\n")
        return EXIT_UNSCORED

    document = result.to_json()
    sys.stdout.write(document)
    sys.stderr.write(f"\n{result.table()}\n")
    if args.save:
        try:
            args.save.parent.mkdir(parents=True, exist_ok=True)
            args.save.write_text(document, encoding="utf-8")
        except OSError as err:
            sys.stderr.write(f"cannot save to {args.save}: {err}\n")
            return EXIT_UNSCORED
    if not args.baseline:
        return 0

    try:
        comparison = compare(result, args.baseline, args.tolerance)
    except EvalError as err:
        sys.stderr.write(f"{err}\n")
        return EXIT_UNSCORED
    for warning in comparison.warnings:
        sys.stderr.write(f"warning: {warning}\n")
    if comparison.regressed:
        sys.stderr.write("REGRESSION\n" + "\n".join(comparison.regressed))
        sys.stderr.write("\n")
        return EXIT_REGRESSION
    return 0


if __name__ == "__main__":
    sys.exit(main())
