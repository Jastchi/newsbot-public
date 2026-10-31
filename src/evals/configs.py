"""The config an eval runs with, and the labelled data it runs on."""

import argparse
from dataclasses import dataclass, field
from pathlib import Path

from django.core.exceptions import ImproperlyConfigured
from django.db import DatabaseError
from pydantic import ValidationError

from evals import datasets
from evals.results import EvalError
from utilities import ConfigNotFoundError, load_config, setup_django
from utilities.models import ConfigModel


def _primary(language: str) -> str:
    """Return the language of a tag: "en-US" and "EN" are both "en"."""
    return language.split("-", maxsplit=1)[0].lower()


@dataclass(frozen=True)
class Subject:
    """
    What an eval runs on.

    Attributes:
        config: Settings the pipeline would use. The default is the
            production defaults.
        key: Name of the config, to tell its results apart; None when
            no config was asked for.
        data_dir: Directory with the labelled data for this config.

    """

    config: ConfigModel = field(default_factory=ConfigModel)
    key: str | None = None
    data_dir: Path = datasets.DATA_DIR

    def load_clustering(self) -> datasets.ClusteringDataset:
        """
        Load the clustering data, refusing another language's.

        Raises:
            EvalError: If the data is invalid or its language differs
                from the config's.

        """
        dataset = datasets.load_clustering(self.data_dir)
        if _primary(self.config.language) != _primary(dataset.language):
            msg = (
                f"{self.key or 'the default config'} is "
                f"{self.config.language!r} but the data in {self.data_dir} "
                f"is {dataset.language!r}; point --data at labelled data "
                "in the config's language"
            )
            raise EvalError(msg)
        return dataset

    def load_faithfulness(
        self,
    ) -> tuple[datasets.ClusteringDataset, datasets.FaithfulnessDataset]:
        """Load faithfulness data and the clustering data it uses."""
        clustering = self.load_clustering()
        return clustering, datasets.load_faithfulness(
            clustering,
            self.data_dir,
        )

    def detail(self) -> dict[str, str]:
        """Return what identifies the config in a saved result."""
        return {"config_key": self.key} if self.key else {}


def _from_database(key: str) -> ConfigModel:
    setup_django()
    try:
        return load_config(key)[0]
    except ConfigNotFoundError as err:
        raise EvalError(str(err)) from err
    except (DatabaseError, ImproperlyConfigured, ValidationError) as err:
        msg = f"cannot read config {key!r} from the database: {err}"
        raise EvalError(msg) from err


def _from_file(path: Path) -> ConfigModel:
    try:
        return ConfigModel.model_validate_json(
            path.read_text(encoding="utf-8"),
        )
    except (OSError, ValidationError) as err:
        msg = f"cannot read a config from {path}: {err}"
        raise EvalError(msg) from err


def from_args(args: argparse.Namespace) -> Subject:
    """
    Build the subject the command line asks for.

    Args:
        args: Parsed arguments with config, config_file and data.

    Returns:
        The config from the database (--config) or from a JSON dump of
        a ConfigModel (--config-file), else the production defaults.

    """
    config, key = ConfigModel(), None
    if args.config:
        config, key = _from_database(args.config), args.config
    elif args.config_file:
        config, key = _from_file(args.config_file), args.config_file.stem
    return Subject(config, key, args.data or datasets.DATA_DIR)
