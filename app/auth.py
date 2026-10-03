import datetime as dt
import uuid

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.config import settings
from app.principals import AuthContext

_bearer = HTTPBearer(auto_error=False)
_ALGO = "HS256"  # dev stub. Week 3: swap to RS256/OIDC (verify with the IdP's JWKS).


def issue_token(user_id: str, tenant_id: str, roles: list[str], groups: list[str]) -> str:
    now = dt.datetime.now(dt.timezone.utc)
    claims = {
        "sub": str(user_id),
        "tid": str(tenant_id),
        "roles": list(roles),
        "groups": list(groups),
        "iss": settings.jwt_issuer,
        "aud": settings.jwt_audience,
        "iat": now,
        "exp": now + dt.timedelta(minutes=settings.jwt_ttl_minutes),
    }
    return jwt.encode(claims, settings.jwt_secret, algorithm=_ALGO)


def _unauthorized(detail: str) -> HTTPException:
    return HTTPException(status.HTTP_401_UNAUTHORIZED, detail, headers={"WWW-Authenticate": "Bearer"})


def get_auth_context(creds: HTTPAuthorizationCredentials | None = Depends(_bearer)) -> AuthContext:
    if creds is None:
        raise _unauthorized("missing bearer token")
    try:
        claims = jwt.decode(
            creds.credentials,
            settings.jwt_secret,
            algorithms=[_ALGO],
            audience=settings.jwt_audience,
            issuer=settings.jwt_issuer,
            options={"require": ["exp", "iat", "sub", "tid", "iss", "aud"]},
        )
        roles, groups = claims.get("roles", []), claims.get("groups", [])
        if not isinstance(roles, list) or not isinstance(groups, list):
            raise ValueError("roles/groups must be lists")
        ctx = AuthContext(
            tenant_id=str(uuid.UUID(claims["tid"])),
            user_id=str(claims["sub"]),
            roles=tuple(roles),
            groups=tuple(groups),
        )
        ctx.principals  # validates every principal; raises ValueError on junk
        return ctx
    except (jwt.PyJWTError, ValueError, KeyError, TypeError):
        raise _unauthorized("invalid token")
