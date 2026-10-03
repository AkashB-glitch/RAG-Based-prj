from alembic import context
from sqlalchemy import create_engine, pool

from app.config import settings

# Migrations run as the OWNER role. The API and worker never use this connection.


def run_offline() -> None:
    context.configure(url=settings.owner_database_url, literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()


def run_online() -> None:
    engine = create_engine(settings.owner_database_url, poolclass=pool.NullPool)
    with engine.connect() as conn:
        context.configure(connection=conn)
        with context.begin_transaction():
            context.run_migrations()


run_offline() if context.is_offline_mode() else run_online()
