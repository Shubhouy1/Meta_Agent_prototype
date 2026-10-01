"""Alembic environment for the MetaAgent API database."""

from logging.config import fileConfig

from alembic import context
from sqlmodel import SQLModel

import backend.app.db.models  # noqa: F401  (registers tables on SQLModel.metadata)
from backend.app.core.config import load_api_settings
from backend.app.db.session import make_engine

config = context.config
if config.config_file_name and config.attributes.get("configure_logger", True):
    fileConfig(config.config_file_name)

target_metadata = SQLModel.metadata


def _database_url() -> str:
    # Set programmatically by run_migrations(); otherwise use the app settings.
    url = config.get_main_option("sqlalchemy.url")
    return url or load_api_settings().resolved_database_url


def run_migrations_offline() -> None:
    context.configure(url=_database_url(), target_metadata=target_metadata,
                      literal_binds=True, render_as_batch=True)
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    engine = make_engine(_database_url())
    with engine.connect() as connection:
        # render_as_batch: SQLite cannot ALTER most constraints in place.
        context.configure(connection=connection, target_metadata=target_metadata, render_as_batch=True)
        with context.begin_transaction():
            context.run_migrations()
    engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
