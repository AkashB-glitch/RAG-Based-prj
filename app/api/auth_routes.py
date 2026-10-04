import uuid
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import text

from app.auth import get_auth_context, issue_token
from app.config import settings
from app.db import app_engine, tenant_only_tx
from app.principals import AuthContext

router = APIRouter(tags=["auth"])

# Standard synthetic demo identities matching the seeded database corpus
SEEDED_USERS = {
    ("acme", "alice@acme.test"): {
        "id": "00000000-0000-0000-0000-000000000001",
        "tenant_id": "00000000-0000-0000-0000-00000000000a",
        "roles": ["employee", "engineer"],
        "groups": ["engineering"],
        "name": "Alice",
        "tenant_slug": "acme",
        "tenant_name": "Acme Corp",
        "persona": "alice",
    },
    ("acme", "bob@acme.test"): {
        "id": "00000000-0000-0000-0000-000000000002",
        "tenant_id": "00000000-0000-0000-0000-00000000000a",
        "roles": ["employee", "hr"],
        "groups": ["payroll"],
        "name": "Bob",
        "tenant_slug": "acme",
        "tenant_name": "Acme Corp",
        "persona": "bob",
    },
    ("acme", "charlie@acme.test"): {
        "id": "00000000-0000-0000-0000-000000000003",
        "tenant_id": "00000000-0000-0000-0000-00000000000a",
        "roles": ["employee", "finance"],
        "groups": ["accounting"],
        "name": "Charlie",
        "tenant_slug": "acme",
        "tenant_name": "Acme Corp",
        "persona": "charlie",
    },
    ("globex", "carol@globex.test"): {
        "id": "00000000-0000-0000-0000-000000000004",
        "tenant_id": "00000000-0000-0000-0000-00000000000b",
        "roles": ["employee", "hr"],
        "groups": [],
        "name": "Carol",
        "tenant_slug": "globex",
        "tenant_name": "Globex Ltd",
        "persona": "carol",
    },
}

PERSONA_MAP = {
    "alice": ("acme", "alice@acme.test"),
    "bob": ("acme", "bob@acme.test"),
    "charlie": ("acme", "charlie@acme.test"),
    "carol": ("globex", "carol@globex.test"),
}


class TokenRequest(BaseModel):
    tenant: str = Field(examples=["acme"])
    email: str = Field(examples=["alice@acme.test"])


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class DemoPersonaRequest(BaseModel):
    persona: str = Field(..., description="'alice', 'bob', 'charlie', or 'carol'")


class DemoAuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: str
    tenant_id: str
    tenant_slug: str
    tenant_name: str
    name: str
    email: str
    roles: list[str]
    groups: list[str]
    principals: list[str]


@router.post("/auth/token", response_model=TokenResponse)
async def dev_login(req: TokenRequest):
    """DEV-ONLY stub identity provider. Issues real cryptographically-signed JWT.
    
    Attempts PostgreSQL authentication first; falls back to verified seeded identities
    when in development/fail-closed guard mode.
    """
    if settings.env != "dev":
        raise HTTPException(404, "Dev authentication disabled in production")
    invalid = HTTPException(401, "invalid credentials")

    user_data = None
    tenant_id = None

    # 1. Attempt PostgreSQL user query if database is online
    try:
        async with app_engine.connect() as conn:
            tenant_id = (
                await conn.execute(text("SELECT id::text FROM tenants WHERE slug = :s"), {"s": req.tenant})
            ).scalar_one_or_none()
        if tenant_id:
            async with tenant_only_tx(tenant_id) as conn:
                db_user = (
                    await conn.execute(
                        text("SELECT id::text AS id, roles, groups FROM users WHERE email = :e"), {"e": req.email}
                    )
                ).mappings().first()
                if db_user:
                    user_data = {
                        "id": db_user["id"],
                        "tenant_id": tenant_id,
                        "roles": list(db_user["roles"]),
                        "groups": list(db_user["groups"]),
                    }
    except Exception:
        # Database connection refused or offline: fall back to seeded dev identities
        pass

    # 2. Check seeded development personas if DB is offline or returned no match
    if user_data is None:
        key = (req.tenant.lower(), req.email.lower())
        seeded = SEEDED_USERS.get(key)
        if seeded:
            user_data = seeded
            tenant_id = seeded["tenant_id"]
        else:
            raise invalid

    token = issue_token(user_data["id"], tenant_id, user_data["roles"], user_data["groups"])
    return TokenResponse(access_token=token)


@router.post("/auth/demo-persona", response_model=DemoAuthResponse)
async def demo_persona_login(req: DemoPersonaRequest):
    """Establishes real authenticated session for demonstration personas.
    
    Generates a cryptographically signed JWT. The frontend cannot bypass or manipulate
    authorization attributes; the backend establishes the authoritative AuthContext.
    """
    p_key = req.persona.lower().strip()
    if p_key not in PERSONA_MAP:
        raise HTTPException(400, f"Unknown persona '{req.persona}'. Supported: alice, bob, charlie, carol")

    tenant_slug, email = PERSONA_MAP[p_key]
    token_resp = await dev_login(TokenRequest(tenant=tenant_slug, email=email))

    seeded = SEEDED_USERS[(tenant_slug, email)]
    principals = sorted(
        [f"user:{seeded['id']}"] + [f"role:{r}" for r in seeded["roles"]] + [f"group:{g}" for g in seeded["groups"]]
    )

    return DemoAuthResponse(
        access_token=token_resp.access_token,
        token_type="bearer",
        user_id=seeded["id"],
        tenant_id=seeded["tenant_id"],
        tenant_slug=tenant_slug,
        tenant_name=seeded["tenant_name"],
        name=seeded["name"],
        email=email,
        roles=seeded["roles"],
        groups=seeded["groups"],
        principals=principals,
    )


@router.get("/me")
async def me(ctx: AuthContext = Depends(get_auth_context)):
    """Validates JWT bearer token and returns cryptographically verified identity context."""
    return {
        "tenant_id": ctx.tenant_id,
        "user_id": ctx.user_id,
        "principals": ctx.principals,
        "roles": list(ctx.roles),
        "groups": list(ctx.groups),
    }
