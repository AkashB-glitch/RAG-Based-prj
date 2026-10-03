import datetime as dt

import jwt
import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials

from app.auth import get_auth_context, issue_token
from app.config import settings
from app.principals import AuthContext, validate_principal

TENANT = "0b6f3a3e-1d2c-4c1a-9a7e-3f2b1c0d9e8f"


def creds(token: str) -> HTTPAuthorizationCredentials:
    return HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)


def test_principals_are_derived_and_sorted():
    ctx = AuthContext(tenant_id=TENANT, user_id="u1", roles=("hr", "employee"), groups=("payroll",))
    assert ctx.principals == ["group:payroll", "role:employee", "role:hr", "user:u1"]


@pytest.mark.parametrize("bad", ["role:hr,role:admin", "admin", "role:", "role:h r", "tenant:x", 42])
def test_invalid_principals_rejected(bad):
    with pytest.raises(ValueError):
        validate_principal(bad)


def test_comma_injection_in_role_claim_is_rejected():
    token = issue_token("u1", TENANT, ["hr,role:ceo"], [])
    with pytest.raises(HTTPException) as e:
        get_auth_context(creds(token))
    assert e.value.status_code == 401


def test_valid_token_roundtrip():
    ctx = get_auth_context(creds(issue_token("u1", TENANT, ["hr"], ["payroll"])))
    assert ctx.tenant_id == TENANT and "role:hr" in ctx.principals


def test_forged_signature_rejected():
    now = dt.datetime.now(dt.timezone.utc)
    forged = jwt.encode(
        {"sub": "u1", "tid": TENANT, "roles": ["hr"], "iss": settings.jwt_issuer, "aud": settings.jwt_audience,
         "iat": now, "exp": now + dt.timedelta(minutes=5)},
        "not-the-secret", algorithm="HS256",
    )
    with pytest.raises(HTTPException):
        get_auth_context(creds(forged))


def test_missing_token_rejected():
    with pytest.raises(HTTPException) as e:
        get_auth_context(None)
    assert e.value.status_code == 401
