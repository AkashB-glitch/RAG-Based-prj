"""Adversarial tests for Authentication, JWT Manipulation, and Principal Injection."""
import datetime as dt
import uuid
import jwt
import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials

from app.auth import get_auth_context, issue_token
from app.config import settings
from app.principals import AuthContext, validate_principal

TENANT_A = "00000000-0000-0000-0000-00000000000a"


def bearer(token: str) -> HTTPAuthorizationCredentials:
    return HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)


# -----------------------------------------------------------------------------
# Attack: Forged JWT Signatures & Algorithm Confusion
# -----------------------------------------------------------------------------
def test_attack_forged_jwt_secret():
    """Attacker signs token with an arbitrary/wrong secret key."""
    now = dt.datetime.now(dt.timezone.utc)
    forged = jwt.encode(
        {
            "sub": "attacker",
            "tid": TENANT_A,
            "roles": ["admin"],
            "groups": [],
            "iss": settings.jwt_issuer,
            "aud": settings.jwt_audience,
            "iat": now,
            "exp": now + dt.timedelta(hours=1),
        },
        "wrong-secret-key-12345678901234567890",
        algorithm="HS256",
    )
    with pytest.raises(HTTPException) as exc:
        get_auth_context(bearer(forged))
    assert exc.value.status_code == 401


def test_attack_expired_jwt_token():
    """Attacker attempts to replay an expired token."""
    now = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=1)
    expired = jwt.encode(
        {
            "sub": "alice",
            "tid": TENANT_A,
            "roles": ["engineer"],
            "groups": [],
            "iss": settings.jwt_issuer,
            "aud": settings.jwt_audience,
            "iat": now - dt.timedelta(hours=1),
            "exp": now,  # expired 24h ago
        },
        settings.jwt_secret,
        algorithm="HS256",
    )
    with pytest.raises(HTTPException) as exc:
        get_auth_context(bearer(expired))
    assert exc.value.status_code == 401


def test_attack_none_algorithm():
    """Attacker attempts algorithm confusion using 'none' algorithm."""
    now = dt.datetime.now(dt.timezone.utc)
    # PyJWT rejects 'none' when algorithm='HS256' is required
    token_none = jwt.encode(
        {
            "sub": "admin",
            "tid": TENANT_A,
            "roles": ["admin"],
            "groups": [],
            "iss": settings.jwt_issuer,
            "aud": settings.jwt_audience,
            "iat": now,
            "exp": now + dt.timedelta(hours=1),
        },
        key="",
        algorithm="none",
    )
    with pytest.raises(HTTPException) as exc:
        get_auth_context(bearer(token_none))
    assert exc.value.status_code == 401


def test_attack_missing_required_claims():
    """Attacker submits token missing mandatory claims (e.g. tid or sub)."""
    now = dt.datetime.now(dt.timezone.utc)
    incomplete = jwt.encode(
        {"roles": ["admin"], "iat": now, "exp": now + dt.timedelta(hours=1)},
        settings.jwt_secret,
        algorithm="HS256",
    )
    with pytest.raises(HTTPException) as exc:
        get_auth_context(bearer(incomplete))
    assert exc.value.status_code == 401


# -----------------------------------------------------------------------------
# Attack: Principal & Delimiter Injection
# -----------------------------------------------------------------------------
@pytest.mark.parametrize(
    "malicious_principal",
    [
        "role:engineer,role:admin",       # Comma injection to escalate privileges
        "user:alice,role:superadmin",      # Comma injection in user ID
        "role:admin; DROP TABLE users;--", # SQL injection in principal
        "role:admin\nrole:ceo",            # Newline injection
        "role:admin\0role:root",           # Null byte injection
        "role:admin' OR '1'='1",           # SQL quote injection
        "role:  admin  ",                  # Whitespace injection
        "admin",                           # Missing prefix
        "super:admin",                     # Invalid prefix
        "",                                # Empty principal
    ],
)
def test_attack_principal_delimiter_and_sql_injection(malicious_principal):
    """Principals must be strictly validated by regex and cannot contain delimiters or SQL."""
    with pytest.raises(ValueError):
        validate_principal(malicious_principal)


def test_attack_jwt_role_claim_injection_rejected():
    """Attacker attempts to forge a role claim with injected comma delimiter."""
    token = issue_token(
        user_id="alice",
        tenant_id=TENANT_A,
        roles=["engineer,role:admin"],
        groups=[],
    )
    with pytest.raises(HTTPException) as exc:
        get_auth_context(bearer(token))
    assert exc.value.status_code == 401


def test_attack_invalid_tenant_uuid_rejected():
    """Attacker submits non-UUID tenant identifier in JWT claim."""
    now = dt.datetime.now(dt.timezone.utc)
    token = jwt.encode(
        {
            "sub": "alice",
            "tid": "not-a-valid-uuid-string",
            "roles": ["engineer"],
            "groups": [],
            "iss": settings.jwt_issuer,
            "aud": settings.jwt_audience,
            "iat": now,
            "exp": now + dt.timedelta(hours=1),
        },
        settings.jwt_secret,
        algorithm="HS256",
    )
    with pytest.raises(HTTPException) as exc:
        get_auth_context(bearer(token))
    assert exc.value.status_code == 401
