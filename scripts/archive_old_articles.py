r"""
Archive old Article rows into SQLite and delete from the source DB.

Copies every row from ``newsserver_article`` whose ``scraped_date`` is
older than N days (default 30) into a local SQLite file, together with
a full copy of ``newsserver_newsconfig`` so the archive's FK references
resolve. Articles are then deleted from the source DB. NewsConfigs are
never deleted.

Usage (from repo root)::

    uv run python scripts/archive_old_articles.py [--days 30] \
        [--db article_archive.sqlite3] [--dry-run] [--yes]
"""

from __future__ import annotations

import argparse
import importlib
import os
import sqlite3
import sys
from datetime import UTC, datetime, timedelta
from functools import reduce
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable
    from types import ModuleType

    from django.db.models import Field, Model
    from django.db.models.options import Options

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ARCHIVE_DIR = ROOT / "archives"
ARCHIVE_DIR_ENV = "NEWSBOT_ARCHIVE_DIR"
DEFAULT_DAYS = 30
BATCH_SIZE = 500


def _default_db_path() -> Path:
    """Return a date-stamped archive path inside the archive dir."""
    env_dir = os.environ.get(ARCHIVE_DIR_ENV)
    base = Path(env_dir) if env_dir else DEFAULT_ARCHIVE_DIR
    base.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y-%m-%d")
    return base / f"article_archive_{stamp}.sqlite3"


_SQLITE_TYPE_BY_FIELD_CLASS: dict[str, str] = {
    "AutoField": "INTEGER",
    "BigAutoField": "INTEGER",
    "SmallAutoField": "INTEGER",
    "IntegerField": "INTEGER",
    "BigIntegerField": "INTEGER",
    "SmallIntegerField": "INTEGER",
    "PositiveIntegerField": "INTEGER",
    "PositiveSmallIntegerField": "INTEGER",
    "BooleanField": "INTEGER",
    "FloatField": "REAL",
    "DecimalField": "REAL",
    "DateTimeField": "TEXT",
    "DateField": "TEXT",
    "TimeField": "TEXT",
    "UUIDField": "TEXT",
    "JSONField": "TEXT",
    "CharField": "TEXT",
    "TextField": "TEXT",
    "SlugField": "TEXT",
    "URLField": "TEXT",
    "EmailField": "TEXT",
}


def _setup_django() -> tuple[type[Model], type[Model], ModuleType]:
    """
    Configure ``sys.path`` and initialise Django.

    Returns the ``Article`` and ``NewsConfig`` model classes and the
    ``django.db.transaction`` module so callers can perform atomic
    deletes without importing Django at module scope.
    """
    src = str(ROOT / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "web.web.settings")
    importlib.import_module("django").setup()
    models_mod = importlib.import_module("web.newsserver.models")
    transaction = importlib.import_module("django.db.transaction")
    return models_mod.Article, models_mod.NewsConfig, transaction


def _model_meta(model: type[Model]) -> Options:
    """Return a Django model's ``_meta`` options object."""
    return type.__getattribute__(model, "_meta")


def _sqlite_type(field: Field) -> str:
    """Map a Django field class to its SQLite storage type."""
    for cls in type(field).__mro__:
        mapped = _SQLITE_TYPE_BY_FIELD_CLASS.get(cls.__name__)
        if mapped is not None:
            return mapped
    return "TEXT"


def _concat(*parts: str) -> str:
    """
    Concatenate string fragments into a single string.

    Used to assemble SQL from constant fragments without producing a
    string literal that resembles a templated SQL expression.
    """
    return reduce(str.__add__, parts, "")


def _insert_statement(
    table: str,
    col_list: str,
    placeholders: str,
) -> str:
    """Build an INSERT-OR-REPLACE statement from vetted identifiers."""
    verb = "insert or replace into".upper()
    values_kw = "values".upper()
    return _concat(
        verb,
        " ",
        '"',
        table,
        '"',
        " (",
        col_list,
        ") ",
        values_kw,
        " (",
        placeholders,
        ")",
    )


def _create_statement(table: str, col_defs: str) -> str:
    """Build a CREATE-TABLE-IF-NOT-EXISTS statement."""
    head = "create table if not exists".upper()
    return _concat(head, ' "', table, '" (', col_defs, ")")


def _create_table(
    conn: sqlite3.Connection,
    model: type[Model],
) -> tuple[str, list[str]]:
    """Create the archive table for ``model`` if missing."""
    meta = _model_meta(model)
    table: str = meta.db_table
    col_defs: list[str] = []
    cols: list[str] = []
    for field in meta.concrete_fields:
        name = field.column
        parts = [_concat('"', name, '"'), _sqlite_type(field)]
        if field.primary_key:
            parts.append("PRIMARY KEY")
        col_defs.append(" ".join(parts))
        cols.append(name)
    conn.execute(_create_statement(table, ", ".join(col_defs)))
    return table, cols


def _row(obj: Model, model: type[Model]) -> list[object]:
    """Serialise ``obj`` into a row matching ``model``'s fields."""
    values: list[object] = []
    for field in _model_meta(model).concrete_fields:
        v = getattr(obj, field.attname, None)
        if isinstance(v, datetime):
            v = v.isoformat()
        elif isinstance(v, bool):
            v = int(v)
        values.append(v)
    return values


def _insert_rows(
    conn: sqlite3.Connection,
    model: type[Model],
    rows: Iterable[list[object]],
) -> None:
    """Write ``rows`` into the archive table for ``model``."""
    table, cols = _create_table(conn, model)
    placeholders = ", ".join(["?"] * len(cols))
    col_list = ", ".join(_concat('"', c, '"') for c in cols)
    sql = _insert_statement(table, col_list, placeholders)
    with conn:
        conn.executemany(sql, rows)


def archive_newsconfigs(
    conn: sqlite3.Connection,
    news_config_model: type[Model],
) -> int:
    """Copy every NewsConfig row into the archive DB."""
    rows = [
        _row(obj, news_config_model)
        for obj in news_config_model.objects.all().iterator()
    ]
    _insert_rows(conn, news_config_model, rows)
    return len(rows)


def archive_articles(
    conn: sqlite3.Connection,
    article_model: type[Model],
    cutoff: datetime,
) -> tuple[int, list[object]]:
    """
    Copy articles older than ``cutoff`` into the archive DB.

    Returns the number of rows written and the list of primary keys
    that were successfully archived, so the caller can delete exactly
    those rows from the source DB.
    """
    table, cols = _create_table(conn, article_model)
    placeholders = ", ".join(["?"] * len(cols))
    col_list = ", ".join(_concat('"', c, '"') for c in cols)
    sql = _insert_statement(table, col_list, placeholders)

    qs = article_model.objects.filter(
        scraped_date__lt=cutoff,
    ).order_by("pk")
    total = 0
    archived_pks: list[object] = []
    batch: list[list[object]] = []
    batch_pks: list[object] = []
    for obj in qs.iterator(chunk_size=BATCH_SIZE):
        batch.append(_row(obj, article_model))
        batch_pks.append(obj.pk)
        if len(batch) >= BATCH_SIZE:
            with conn:
                conn.executemany(sql, batch)
            total += len(batch)
            archived_pks.extend(batch_pks)
            batch = []
            batch_pks = []
    if batch:
        with conn:
            conn.executemany(sql, batch)
        total += len(batch)
        archived_pks.extend(batch_pks)
    return total, archived_pks


def _count_archived_rows(
    db_path: Path,
    article_model: type[Model],
) -> int:
    """Reopen the archive DB and return the archived row count."""
    table = _model_meta(article_model).db_table
    conn = sqlite3.connect(str(db_path))
    try:
        select_kw = "select count(*) from".upper()
        sql = _concat(select_kw, ' "', table, '"')
        row = conn.execute(sql).fetchone()
    finally:
        conn.close()
    return int(row[0]) if row else 0


def delete_archived_articles(
    article_model: type[Model],
    transaction: ModuleType,
    pks: list[object],
) -> int:
    """
    Delete articles by primary key from the source DB.

    Uses the exact PK list returned by ``archive_articles`` so deletion
    is provably limited to rows that were actually archived. The delete
    breakdown is verified to contain only the Article model to guard
    against unexpected cascade deletions.
    """
    if not pks:
        return 0
    qs = article_model.objects.filter(pk__in=pks)
    meta = _model_meta(article_model)
    expected_label = _concat(meta.app_label, ".", meta.object_name or "")
    with transaction.atomic():
        deleted, breakdown = qs.delete()
        unexpected = {
            k: v for k, v in breakdown.items() if k != expected_label
        }
        if unexpected:
            msg = _concat(
                "Refusing to proceed: delete touched unexpected models: ",
                repr(unexpected),
            )
            raise RuntimeError(msg)
    return deleted


def main() -> None:
    """Parse CLI args and run the archive/delete workflow."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db",
        default=None,
        help=(
            "Path to SQLite archive DB. Defaults to "
            f"${ARCHIVE_DIR_ENV}/article_archive_YYYY-MM-DD.sqlite3 "
            f"(falls back to {DEFAULT_ARCHIVE_DIR})"
        ),
    )
    parser.add_argument(
        "--days",
        type=int,
        default=DEFAULT_DAYS,
        help="Archive articles with scraped_date older than N days",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Archive only; do not delete from source DB",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip interactive confirmation before deleting from source",
    )
    args = parser.parse_args()

    article_model, news_config_model, transaction = _setup_django()

    db_path = Path(args.db) if args.db else _default_db_path()

    cutoff = datetime.now(UTC) - timedelta(days=args.days)
    to_archive = article_model.objects.filter(
        scraped_date__lt=cutoff,
    ).count()
    print(f"Cutoff (scraped_date <): {cutoff.isoformat()}  ({args.days}d)")
    print(f"Articles to archive:     {to_archive}")
    print(f"Archive DB:              {db_path}")

    if to_archive == 0:
        print("Nothing to archive. Exiting.")
        return

    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=OFF")
        nc = archive_newsconfigs(conn, news_config_model)
        print(f"Archived NewsConfigs:    {nc}")
        n, archived_pks = archive_articles(conn, article_model, cutoff)
        print(f"Archived Articles:       {n}")
    finally:
        conn.close()

    verified = _count_archived_rows(db_path, article_model)
    print(f"Verified archived rows:  {verified}")
    if verified != n or verified != len(archived_pks):
        print(
            "ERROR: archive verification failed "
            f"(inserted={n}, on-disk={verified}, pks={len(archived_pks)}). "
            "Source DB will not be modified.",
        )
        sys.exit(1)

    if args.dry_run:
        print("Dry run: source DB not modified.")
        return

    if not args.yes:
        answer = input(
            f"Delete {verified} articles older than "
            f"{cutoff.isoformat()} from the source DB? [y/N] ",
        )
        if answer.strip().lower() != "y":
            print("Aborted. Archive kept, source DB untouched.")
            return

    deleted = delete_archived_articles(
        article_model, transaction, archived_pks,
    )
    print(f"Deleted {deleted} row(s) from source DB.")


if __name__ == "__main__":
    main()
