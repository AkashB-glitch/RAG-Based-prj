"""Every query the API makes about tenant data goes through tenant_tx().

It opens a transaction and sets the RLS context with set_config(..., is_local=true), so the
settings vanish at COMMIT/ROLLBACK and can never leak to the next request on a pooled
connection. This also works behind transaction-mode poolers (e.g. Supabase/PgBouncer).
"""
from contextlib import asynccontextmanager
from typing import AsyncIterator

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection, create_async_engine

from app.config import settings
from app.principals import AuthContext

app_engine = create_async_engine(settings.app_database_url, pool_pre_ping=True, pool_size=10)

_SET_CTX = text(
    "SELECT set_config('app.tenant_id', :tenant_id, true),"
    "       set_config('app.user_id', :user_id, true),"
    "       set_config('app.principals', :principals, true)"
)


@asynccontextmanager
async def tenant_tx(ctx: AuthContext) -> AsyncIterator[AsyncConnection]:
    async with app_engine.begin() as conn:
        await conn.execute(
            _SET_CTX,
            {"tenant_id": ctx.tenant_id, "user_id": ctx.user_id, "principals": ctx.principals_csv},
        )
        yield conn


@asynccontextmanager
async def tenant_only_tx(tenant_id: str) -> AsyncIterator[AsyncConnection]:
    """Tenant context with NO principals. Used only by the dev login to look up a user row."""
    async with app_engine.begin() as conn:
        await conn.execute(text("SELECT set_config('app.tenant_id', :t, true)"), {"t": tenant_id})
        yield conn
