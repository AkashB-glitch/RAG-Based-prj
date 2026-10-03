from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import text

from app.auth import get_auth_context, issue_token
from app.config import settings
from app.db import app_engine, tenant_only_tx
from app.principals import AuthContext

router = APIRouter(tags=["auth"])


class TokenRequest(BaseModel):
    tenant: str = Field(examples=["acme"])
    email: str = Field(examples=["alice@acme.test"])


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


@router.post("/auth/token", response_model=TokenResponse)
async def dev_login(req: TokenRequest):
    """DEV-ONLY stub identity provider: no password. Disabled unless ENV=dev."""
    if settings.env != "dev":
        raise HTTPException(404)
    invalid = HTTPException(401, "invalid credentials")

    async with app_engine.connect() as conn:
        tenant_id = (
            await conn.execute(text("SELECT id::text FROM tenants WHERE slug = :s"), {"s": req.tenant})
        ).scalar_one_or_none()
    if tenant_id is None:
        raise invalid

    # users has RLS: it is only readable once a tenant is set, so a login for tenant A
    # can never resolve a user row from tenant B.
    async with tenant_only_tx(tenant_id) as conn:
        user = (
            await conn.execute(
                text("SELECT id::text AS id, roles, groups FROM users WHERE email = :e"), {"e": req.email}
            )
        ).mappings().first()
    if user is None:
        raise invalid
    return TokenResponse(access_token=issue_token(user["id"], tenant_id, user["roles"], user["groups"]))


@router.get("/me")
async def me(ctx: AuthContext = Depends(get_auth_context)):
    return {"tenant_id": ctx.tenant_id, "user_id": ctx.user_id, "principals": ctx.principals}
