"""Tests for the evaluations: metrics, datasets, results and the CLI."""

import argparse
import importlib
import inspect
import json
import pkgutil
import random
import shutil
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import pytest
from django.core.exceptions import ImproperlyConfigured
from pydantic import ValidationError
from sklearn.metrics import cohen_kappa_score

import newsbot.agents
from evals import clustering, configs, faithfulness, metrics
from evals.__main__ import main
from evals.datasets import (
    BASELINES_DIR,
    DATA_DIR,
    ClusteringDataset,
    FaithfulnessCase,
    load_clustering,
    load_faithfulness,
)
from evals.offline import without_llm_provider
from evals.results import EvalError, Result, compare
from newsbot.agents.name_validation_agent import NameValidationAgent
from newsbot.agents.story_clustering_agent import StoryClusteringAgent
from newsbot.agents.summarization_agent import SummarizationAgent
from newsbot.model_cache import get_sentence_transformer
from newsbot.models import JudgeVerdict, ValidationStats
from utilities import ConfigNotFoundError
from utilities.models import (
    ConfigModel,
    LLMConfigModel,
    StoryClusteringConfigModel,
)


@pytest.fixture(autouse=True, scope="module")
def no_llm_server():
    """
    Make building any agent's provider fail, as on a runner w/o Ollama.

    Every agent module is found by scanning the package, not from the
    list in evals.offline, so an agent missing from that list fails.
    """

    def unreachable(config):
        msg = "no LLM server in the tests"
        raise ConnectionError(msg)

    modules = (
        importlib.import_module(f"newsbot.agents.{info.name}")
        for info in pkgutil.iter_modules(newsbot.agents.__path__)
    )
    with pytest.MonkeyPatch.context() as patch:
        for module in modules:
            if hasattr(module, "get_llm_provider"):
                patch.setattr(module, "get_llm_provider", unreachable)
        yield


class TestWithoutLlmProvider:
    def test_agents_are_built_without_a_server_and_cannot_chat(self):
        config = faithfulness._config(None, ConfigModel())
        with pytest.raises(ConnectionError):
            NameValidationAgent(config)
        with without_llm_provider():
            agent = NameValidationAgent(config)
        with pytest.raises(RuntimeError, match="must not call an LLM"):
            agent.provider.chat([], {})

    def test_the_real_factories_are_back_afterwards(self):
        with without_llm_provider():
            pass
        with pytest.raises(ConnectionError):
            NameValidationAgent(faithfulness._config(None, ConfigModel()))


class TestClusteringScores:
    def test_perfect_clustering_scores_one(self):
        truth = ["a", "a", "b", "b", None]
        scores = metrics.clustering_scores(truth, [[0, 1], [2, 3]])
        assert scores.ari == pytest.approx(1.0)
        assert scores.pair_f1 == pytest.approx(1.0)

    def test_merged_stories_lose_precision_not_recall(self):
        truth = ["a", "a", "b", "b"]
        scores = metrics.clustering_scores(truth, [[0, 1, 2, 3]])
        assert scores.pair_recall == pytest.approx(1.0)
        assert scores.pair_precision == pytest.approx(1 / 3)
        assert scores.ari < 1.0

    def test_unclustered_articles_count_as_missed_pairs(self):
        truth = ["a", "a", "b", "b"]
        scores = metrics.clustering_scores(truth, [[0, 1]])
        assert scores.pair_precision == pytest.approx(1.0)
        assert scores.pair_recall == pytest.approx(0.5)

    def test_predicting_no_pairs_scores_zero(self):
        scores = metrics.clustering_scores(["a", "a"], [])
        assert scores.pair_precision == 0.0
        assert scores.pair_recall == 0.0

    def test_nothing_to_score_is_zero_not_a_crash(self):
        scores = metrics.clustering_scores([], [])
        assert scores.as_metrics() == {
            "ari": 1.0,
            "pair_precision": 0.0,
            "pair_recall": 0.0,
            "pair_f1": 0.0,
        }

    @pytest.mark.parametrize("bad", [[[0, 5]], [[-1, 0]]])
    def test_an_unknown_article_index_is_an_error(self, bad):
        with pytest.raises(ValueError, match="outside the labels"):
            metrics.clustering_scores(["a", "a"], bad)

    def test_an_article_in_two_clusters_is_an_error(self):
        with pytest.raises(ValueError, match="more than one cluster"):
            metrics.clustering_scores(["a", "a", "b"], [[0, 1], [1, 2]])


class TestDetectionScores:
    def test_counts_and_rates(self):
        scores = metrics.DetectionScores.from_flags(
            actual=[True, True, False, False],
            flagged=[True, False, True, False],
        )
        assert scores.counts() == {"tp": 1, "fp": 1, "fn": 1, "tn": 1}
        assert scores.precision == pytest.approx(0.5)
        assert scores.specificity == pytest.approx(0.5)
        assert scores.kappa == pytest.approx(0.0)

    def test_kappa_matches_scikit_learn(self):
        rng = random.Random(7)
        for _ in range(20):
            actual = [rng.random() < 0.5 for _ in range(30)]
            flagged = [rng.random() < 0.4 for _ in range(30)]
            scores = metrics.DetectionScores.from_flags(actual, flagged)
            assert scores.kappa == pytest.approx(
                cohen_kappa_score(actual, flagged),
            )

    def test_perfect_detector_has_kappa_one(self):
        flags = [True, False, True, False]
        scores = metrics.DetectionScores.from_flags(flags, flags)
        assert scores.kappa == 1.0

    def test_detector_that_flags_nothing_is_not_perfect(self):
        scores = metrics.DetectionScores.from_flags([True, True], [False] * 2)
        assert scores.kappa == 0.0

    def test_no_cases_scores_zero_without_dividing_by_zero(self):
        assert metrics.DetectionScores.from_flags([], []).as_metrics() == {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "specificity": 0.0,
            "accuracy": 0.0,
            "kappa": 0.0,
        }


class TestUngroundedFacts:
    SOURCE = "The Fed held rates at 4.25% on Wednesday. About 3,000 left."

    def test_grounded_summary_has_no_findings(self):
        summary = "Rates stayed at 4.25% on Wednesday; 3000 left."
        assert not metrics.ungrounded_facts(summary, self.SOURCE)

    @pytest.mark.parametrize(
        ("summary", "expected"),
        [
            ("Rates stayed at 4.00%.", {"4"}),
            ("The Fed met on Friday.", {"friday"}),
            ("About 30,000 left.", {"30000"}),
        ],
    )
    def test_changed_figures_and_dates_are_found(self, summary, expected):
        assert metrics.ungrounded_facts(summary, self.SOURCE) == expected

    def test_dotted_dates_do_not_crash(self):
        found = metrics.ungrounded_facts("On 12.03.2025", "x")
        assert found == {"12.03", "2025"}

    def test_large_figures_are_compared_exactly(self):
        found = metrics.ungrounded_facts("1,234,567 died", "1,234,568 died")
        assert found == {"1234567"}

    def test_scale_words_match_written_out_figures(self):
        assert not metrics.ungrounded_facts(
            "a 1.5 million loss",
            "a loss of 1,500,000",
        )
        assert metrics.ungrounded_facts("a 2 million loss", "1,500,000")
        assert not metrics.ungrounded_facts("a 1.5-million loss", "1500000")

    def test_a_scale_word_on_the_next_line_is_not_a_scale(self):
        assert not metrics.ungrounded_facts("in 2025\nMillion more", "2025")

    def test_second_and_march_are_not_facts(self):
        assert not metrics.ungrounded_facts(
            "a second round as troops march",
            "another round",
        )

    def test_spelled_out_numbers_match_digits(self):
        assert not metrics.ungrounded_facts("twelve jets", "12 jets flew")
        assert metrics.ungrounded_facts("twenty jets", "12 jets flew")


class TestRegressions:
    def test_only_drops_beyond_tolerance_are_reported(self):
        baseline = {"ari": 0.8, "f1": 0.9}
        current = {"ari": 0.7, "f1": 0.88, "new": 0.1}
        assert metrics.regressions(current, baseline, 0.05) == [
            "ari: 0.800 -> 0.700",
        ]

    def test_a_drop_of_exactly_the_tolerance_passes(self):
        # powers of two, so the subtraction is exact
        assert not metrics.regressions({"ari": 0.75}, {"ari": 1.0}, 0.25)
        assert metrics.regressions({"ari": 0.7}, {"ari": 1.0}, 0.25)

    def test_metric_missing_from_current_run_is_reported(self):
        assert metrics.regressions({}, {"judge.f1": 0.9}) == [
            "judge.f1: 0.900 -> missing",
        ]


class TestDatasets:
    def test_committed_data_loads_and_is_linked(self):
        data = load_clustering()
        cases = load_faithfulness(data).cases
        assert data.stories()
        assert any(c.is_unfaithful for c in cases)
        assert any(not c.is_unfaithful for c in cases)

    def test_a_story_needs_two_articles(self):
        row = {"source": "s", "title": "t", "content": "c"}
        with pytest.raises(ValidationError, match="single article"):
            ClusteringDataset.model_validate(
                {
                    "description": "",
                    "articles": [{"id": "a", "story": "x", **row}],
                },
            )

    def test_a_misspelt_key_is_an_error(self):
        with pytest.raises(ValidationError, match="stroy"):
            ClusteringDataset.model_validate(
                {
                    "description": "",
                    "articles": [
                        {
                            "id": "a",
                            "stroy": "x",
                            "source": "s",
                            "title": "t",
                            "content": "c",
                        },
                    ],
                },
            )

    def test_error_type_is_set_exactly_for_unfaithful_cases(self):
        base = {"id": "f", "story": "x", "summary": "s"}
        with pytest.raises(ValidationError, match="exactly when"):
            FaithfulnessCase(**base, label="faithful", error="invented")
        with pytest.raises(ValidationError, match="exactly when"):
            FaithfulnessCase(**base, label="unfaithful", error=None)


class TestResults:
    @staticmethod
    def _saved(tmp_path: Path, scores: dict, config: dict | None) -> Path:
        path = tmp_path / "baseline.json"
        path.write_text(
            json.dumps({"metrics": scores, "detail": {"config": config}}),
        )
        return path

    def test_compare_reports_regression_and_changed_settings(self, tmp_path):
        baseline = self._saved(tmp_path, {"ari": 0.9}, {"threshold": 0.7})
        result = Result("c", {"ari": 0.5}, {"config": {"threshold": 1}})
        comparison = compare(result, baseline, tolerance=0.05)
        assert comparison.regressed == ["ari: 0.900 -> 0.500"]
        assert len(comparison.warnings) == 1

    def test_compare_passes_an_equal_run(self, tmp_path):
        baseline = self._saved(tmp_path, {"ari": 0.9}, {"threshold": 0.7})
        result = Result("c", {"ari": 0.9}, {"config": {"threshold": 0.7}})
        comparison = compare(result, baseline, tolerance=0.05)
        assert not comparison.regressed
        assert not comparison.warnings

    @pytest.mark.parametrize("content", ["not json", "{}", "[]"])
    def test_unreadable_baseline_is_an_eval_error(self, tmp_path, content):
        path = tmp_path / "baseline.json"
        path.write_text(content)
        with pytest.raises(EvalError, match="cannot read a baseline"):
            compare(Result("c", {}), path, tolerance=0.05)

    def test_missing_baseline_is_an_eval_error(self, tmp_path):
        with pytest.raises(EvalError, match="cannot read a baseline"):
            compare(Result("c", {}), tmp_path / "nope.json", tolerance=0.05)

    def test_skipped_prefixes_hide_only_their_own_metrics(self, tmp_path):
        baseline = self._saved(
            tmp_path,
            {"judge.f1": 1.0, "numbers.f1": 0.9},
            None,
        )
        comparison = compare(
            Result("f", {"numbers.f1": 0.5}, unscored=("judge.",)),
            baseline,
            tolerance=0.05,
        )
        assert comparison.regressed == ["numbers.f1: 0.900 -> 0.500"]
        assert "1 baseline metric(s) starting judge." in comparison.warnings[0]

    def test_a_baseline_with_nothing_left_to_compare_is_an_error(
        self,
        tmp_path,
    ):
        baseline = self._saved(tmp_path, {"judge.f1": 1.0}, None)
        result = Result("f", {"numbers.f1": 0.9}, unscored=("judge.",))
        with pytest.raises(EvalError, match="nothing to compare"):
            compare(result, baseline, tolerance=0.05)

    def test_compare_warns_when_the_config_differs(self, tmp_path):
        path = tmp_path / "baseline.json"
        path.write_text(
            json.dumps(
                {"metrics": {"ari": 0.9}, "detail": {"config_key": "a"}},
            ),
        )
        result = Result("c", {"ari": 0.9}, {"config_key": "b"})
        assert len(compare(result, path, tolerance=0.05).warnings) == 1

    def test_table_aligns_metric_names(self):
        table = Result("x", {"a": 1.0, "long_name": 0.5}).table()
        assert table.splitlines() == ["a          1.000", "long_name  0.500"]


class TestCli:
    @staticmethod
    def _stub_clustering(monkeypatch, ari: float):
        monkeypatch.setattr(
            clustering,
            "run",
            lambda overrides, subject: Result(
                "clustering",
                {"ari": ari},
                {"config": dict(overrides), **subject.detail()},
            ),
        )

    def test_judge_needs_yes_and_says_what_it_costs(self, capsys):
        assert main(["faithfulness", "--judge"]) == 2
        err = capsys.readouterr().err
        assert "billed" in err
        assert "--yes" in err

    def test_save_then_baseline_passes(self, tmp_path, monkeypatch, capsys):
        self._stub_clustering(monkeypatch, ari=0.8)
        saved = tmp_path / "run.json"
        assert main(["clustering", "--save", str(saved)]) == 0
        assert json.loads(saved.read_text())["metrics"] == {"ari": 0.8}
        assert main(["clustering", "--baseline", str(saved)]) == 0
        capsys.readouterr()

    def test_save_creates_missing_directories(
        self,
        tmp_path,
        monkeypatch,
        capsys,
    ):
        self._stub_clustering(monkeypatch, ari=0.8)
        saved = tmp_path / "new" / "dir" / "run.json"
        assert main(["clustering", "--save", str(saved)]) == 0
        assert saved.exists()
        capsys.readouterr()

    def test_an_unwritable_save_path_exits_two(
        self,
        tmp_path,
        monkeypatch,
        capsys,
    ):
        self._stub_clustering(monkeypatch, ari=0.8)
        blocker = tmp_path / "file"
        blocker.write_text("")
        assert main(["clustering", "--save", str(blocker / "run.json")]) == 2
        assert "cannot save" in capsys.readouterr().err

    def test_a_drop_against_the_baseline_exits_one(
        self,
        tmp_path,
        monkeypatch,
        capsys,
    ):
        self._stub_clustering(monkeypatch, ari=0.8)
        saved = tmp_path / "run.json"
        main(["clustering", "--save", str(saved)])
        self._stub_clustering(monkeypatch, ari=0.4)
        assert main(["clustering", "--baseline", str(saved)]) == 1
        assert "ari: 0.800 -> 0.400" in capsys.readouterr().err

    def test_a_bad_baseline_exits_two_not_one(self, tmp_path, monkeypatch):
        self._stub_clustering(monkeypatch, ari=0.8)
        baseline = tmp_path / "bad.json"
        baseline.write_text("{}")
        assert main(["clustering", "--baseline", str(baseline)]) == 2

    @staticmethod
    def _config_file(tmp_path: Path, **fields) -> Path:
        path = tmp_path / "tech.json"
        path.write_text(ConfigModel(**fields).model_dump_json())
        return path

    def test_a_config_file_reaches_the_run_and_names_the_result(
        self,
        tmp_path,
        monkeypatch,
        capsys,
    ):
        seen = []

        def run(overrides, subject):
            seen.append(subject)
            return Result("clustering", {"ari": 1.0}, subject.detail())

        monkeypatch.setattr(clustering, "run", run)
        path = self._config_file(
            tmp_path,
            story_clustering={"similarity_threshold": 0.55},
        )
        assert main(["clustering", "--config-file", str(path)]) == 0
        assert seen[0].key == "tech"
        assert seen[0].config.story_clustering.similarity_threshold == 0.55
        assert '"config_key": "tech"' in capsys.readouterr().out

    def test_a_config_from_the_database_is_used_by_key(self, monkeypatch):
        monkeypatch.setattr(configs, "setup_django", lambda: None)
        monkeypatch.setattr(
            configs,
            "load_config",
            lambda key: (ConfigModel(name=key), None),
        )
        subject = configs.from_args(
            argparse.Namespace(config="tech", config_file=None, data=None),
        )
        assert (subject.key, subject.config.name) == ("tech", "tech")

    @pytest.mark.parametrize("error", ["django", "invalid"])
    def test_a_database_that_cannot_be_read_exits_two(
        self,
        monkeypatch,
        capsys,
        error,
    ):
        def broken(key):
            if error == "django":
                raise ImproperlyConfigured
            ConfigModel.model_validate({"language": 1})

        monkeypatch.setattr(configs, "setup_django", lambda: None)
        monkeypatch.setattr(configs, "load_config", broken)
        assert main(["clustering", "--config", "tech"]) == 2
        assert "cannot read config 'tech'" in capsys.readouterr().err

    def test_an_unknown_database_config_exits_two(self, monkeypatch, capsys):
        def missing(key):
            msg = f"Config not found: {key}"
            raise ConfigNotFoundError(msg)

        monkeypatch.setattr(configs, "setup_django", lambda: None)
        monkeypatch.setattr(configs, "load_config", missing)
        assert main(["clustering", "--config", "nope"]) == 2
        assert "Config not found: nope" in capsys.readouterr().err

    def test_an_invalid_config_file_exits_two(self, tmp_path, capsys):
        bad = tmp_path / "bad.json"
        bad.write_text('{"story_clustering": {"top_stories_count": "many"}}')
        assert main(["clustering", "--config-file", str(bad)]) == 2
        assert "cannot read a config" in capsys.readouterr().err

    def test_a_config_in_another_language_is_not_scored_on_english_data(
        self,
        tmp_path,
        capsys,
    ):
        path = self._config_file(tmp_path, language="de")
        assert main(["clustering", "--config-file", str(path)]) == 2
        assert "labelled data in the config's language" in (
            capsys.readouterr().err
        )

    def test_a_missing_data_directory_exits_two(self, tmp_path, capsys):
        assert main(["clustering", "--data", str(tmp_path / "none")]) == 2
        assert "cannot read clustering.json" in capsys.readouterr().err

    def test_the_judge_comes_from_the_config_unless_a_flag_overrides(
        self,
        tmp_path,
        monkeypatch,
    ):
        judges = []

        def run(judge=None, subject=None):
            judges.append(judge)
            return Result("faithfulness", {"judge.f1": 1.0})

        monkeypatch.setattr(faithfulness, "run", run)
        path = self._config_file(
            tmp_path,
            llm=LLMConfigModel(provider="ollama", judge_model="llama"),
        )
        base = ["faithfulness", "--judge", "--yes", "--config-file", str(path)]
        assert main(base) == 0
        assert main([*base, "--judge-model", "other"]) == 0
        assert main(["faithfulness", "--judge", "--yes"]) == 0
        assert [(j.provider, j.model) for j in judges] == [
            ("ollama", "llama"),
            ("ollama", "other"),
            ("gemini", "gemini-2.5-flash"),
        ]

    def test_options_reach_the_clustering_run(self, monkeypatch, capsys):
        self._stub_clustering(monkeypatch, ari=1.0)
        main(["clustering", "--similarity-threshold", "0.5"])
        assert '"similarity_threshold": 0.5' in capsys.readouterr().out


class TestFaithfulnessRun:
    """Runs the free, offline detectors on the committed seed data."""

    @pytest.fixture(scope="class")
    def offline_run(self):
        return faithfulness.run()

    def test_number_and_date_changes_are_caught_without_false_alarms(
        self,
        offline_run,
    ):
        result = offline_run
        recall = result.detail["numbers_dates_recall_by_error"]
        assert recall["changed_number"] == 1.0
        assert recall["changed_date"] == 1.0
        assert result.metrics["deterministic.specificity"] == 1.0

    def test_the_committed_baseline_holds_the_judge_metrics(self):
        saved = json.loads(
            (BASELINES_DIR / "faithfulness.json").read_text(encoding="utf-8"),
        )
        assert any(k.startswith("judge.") for k in saved["metrics"])

    def test_the_free_run_passes_the_committed_baseline_from_the_cli(
        self,
        capsys,
    ):
        baseline = str(BASELINES_DIR / "faithfulness.json")
        argv = ["faithfulness", "--baseline", baseline, "--tolerance", "1e-6"]
        assert main(argv) == 0
        assert "left out of the comparison" in capsys.readouterr().err

    @staticmethod
    def _copy_data(tmp_path: Path, **clustering_fields) -> Path:
        data = tmp_path / "data"
        shutil.copytree(DATA_DIR, data)
        if clustering_fields:
            path = data / "clustering.json"
            path.write_text(
                json.dumps(json.loads(path.read_text()) | clustering_fields),
            )
        return data

    def test_a_data_directory_of_its_own_is_scored(
        self,
        tmp_path,
        offline_run,
    ):
        subject = configs.Subject(data_dir=self._copy_data(tmp_path))
        assert faithfulness.run(subject=subject).metrics == offline_run.metrics

    def test_a_config_in_the_datas_language_is_scored(self, tmp_path):
        subject = configs.Subject(
            ConfigModel(language="de"),
            "de",
            self._copy_data(tmp_path, language="de"),
        )
        result = faithfulness.run(subject=subject)
        assert result.detail["config_key"] == "de"

    @pytest.mark.parametrize("language", ["en-US", "EN"])
    def test_a_language_tag_variant_matches_the_data(self, language):
        subject = configs.Subject(ConfigModel(language=language))
        assert subject.load_clustering().language == "en"

    def test_a_config_run_records_what_it_can_change(self):
        subject = configs.Subject(
            ConfigModel(llm=LLMConfigModel(spacy_model="en_core_web_sm")),
            "tech",
        )
        detail = faithfulness.run(subject=subject).detail
        assert detail["config"] == {"spacy_model": "en_core_web_sm"}

    def test_a_missing_spacy_model_is_an_eval_error(self):
        subject = configs.Subject(
            ConfigModel(llm=LLMConfigModel(spacy_model="xx_not_installed")),
        )
        with pytest.raises(EvalError, match="xx_not_installed"):
            faithfulness.run(subject=subject)

    def test_every_detector_reports_the_same_metrics(self, offline_run):
        names = {k.split(".")[1] for k in offline_run.metrics}
        assert names == {
            "precision",
            "recall",
            "f1",
            "specificity",
            "accuracy",
            "kappa",
        }


class TestJudgeDetector:
    @staticmethod
    def _summarizer(
        stats: ValidationStats,
        calls: list[str],
        rejects: Callable[[str], bool] = lambda summary: summary == "bad",
        **members,
    ):
        def validate_output(summary, context, *, source):
            calls.append(summary)
            stats.inc("judge.checked")
            if summary == "outage":
                stats.inc("judge.fail_open_error")
            return JudgeVerdict(is_valid=not rejects(summary))

        judge = SimpleNamespace(
            validate_output=validate_output,
            stats=stats,
            judge_model="m",
        )
        return SimpleNamespace(
            **{
                "judge_agent": judge,
                "_build_judge_source": lambda articles: "source",
                **members,
            },
        )

    @staticmethod
    def _case(summary: str) -> FaithfulnessCase:
        return FaithfulnessCase(
            id="f",
            story="s",
            label="faithful",
            error=None,
            summary=summary,
        )

    def test_flags_what_the_judge_rejects(self):
        detail: dict = {}
        detector = faithfulness._judge_detector(
            self._summarizer(ValidationStats(), []),
            detail,
        )
        assert detector(self._case("bad"), []) is True
        assert detector(self._case("fine"), []) is False
        assert detail["judge_model"] == "m"

    def test_a_scored_judge_run_reports_metrics_and_counters(
        self,
        monkeypatch,
    ):
        # a judge that agrees with every label
        unfaithful = {
            c.summary: c.is_unfaithful
            for c in load_faithfulness(load_clustering()).cases
        }
        stats = ValidationStats()
        real = object.__new__(SummarizationAgent)
        fake = self._summarizer(
            stats,
            [],
            rejects=unfaithful.__getitem__,
            _build_judge_source=real._build_judge_source,
            _get_article_contents=real._get_article_contents,
        )
        monkeypatch.setattr(faithfulness, "SummarizationAgent", lambda _: fake)
        result = faithfulness.run(faithfulness.JudgeOptions("ollama", "m"))
        assert result.metrics["judge.recall"] == 1.0
        assert result.metrics["judge.kappa"] == 1.0
        assert result.detail["judge_model"] == "m"
        assert result.detail["judge_counters"]["judge.checked"] == len(
            unfaithful,
        )

    def test_a_judge_that_fails_open_stops_the_run(self):
        calls: list[str] = []
        detector = faithfulness._judge_detector(
            self._summarizer(ValidationStats(), calls),
            {},
        )
        with pytest.raises(EvalError, match="failed open"):
            detector(self._case("outage"), [])
        assert calls == ["outage"]


@pytest.fixture(scope="module")
def embedding_model_available():
    model = StoryClusteringConfigModel().embedding_model
    try:
        get_sentence_transformer(model)
    except Exception:
        pytest.skip(f"{model} is not available offline")


@pytest.mark.usefixtures("embedding_model_available")
class TestClusteringRun:
    def test_a_looser_threshold_beats_an_unreachable_one(self):
        loose = clustering.run({"similarity_threshold": 0.6})
        strict = clustering.run({"similarity_threshold": 0.95})
        assert loose.metrics["ari"] > strict.metrics["ari"]
        assert strict.detail["unclustered"] == len(load_clustering().articles)

    def test_the_configs_own_settings_are_used(self):
        subject = configs.Subject(
            ConfigModel(
                story_clustering=StoryClusteringConfigModel(
                    similarity_threshold=0.6,
                ),
            ),
            "tech",
        )
        own = clustering.run({}, subject)
        assert own.detail["config"]["similarity_threshold"] == 0.6
        assert own.detail["config_key"] == "tech"
        assert (
            own.metrics
            == clustering.run(
                {"similarity_threshold": 0.6},
            ).metrics
        )

    def test_a_flag_beats_the_configs_setting(self):
        subject = configs.Subject(
            ConfigModel(
                story_clustering=StoryClusteringConfigModel(
                    similarity_threshold=0.6,
                ),
            ),
        )
        result = clustering.run({"similarity_threshold": 0.95}, subject)
        assert result.detail["config"]["similarity_threshold"] == 0.95

    def test_the_run_reports_its_settings(self):
        result = clustering.run({"algorithm": "greedy"})
        assert result.detail["config"]["algorithm"] == "greedy"


class TestProductionDrift:
    """Fail when production changes what the evals copy or call."""

    def test_judge_context_matches_the_pipeline_prompt(self):
        source = inspect.getsource(SummarizationAgent._build_story_summary)
        literals = "".join(
            line.strip().strip('"')
            for line in source.splitlines()
            if line.strip().startswith('"')
        )
        assert faithfulness.STORY_SUMMARY_CONTEXT in literals

    @pytest.mark.parametrize(
        ("agent", "method"),
        [
            (StoryClusteringAgent, "_get_embedding_texts"),
            (StoryClusteringAgent, "_get_article_locations"),
            (StoryClusteringAgent, "_apply_geo_penalty_to_similarity"),
            (StoryClusteringAgent, "_cluster_articles"),
            (SummarizationAgent, "_build_judge_source"),
            (SummarizationAgent, "_get_article_contents"),
        ],
    )
    def test_agent_methods_the_evals_call_exist(self, agent, method):
        assert callable(getattr(agent, method))
