"""Principals are the strings that ACLs are written in: user:<id>, role:<name>, group:<name>.

They are always derived server-side from a *verified* token, never from the request body.
Principals are not tenant-prefixed on purpose: every row also carries tenant_id, and both the
app filter and the RLS policy require tenant_id to match, so "role:hr" in tenant A can never
match a row in tenant B.
"""
import re
from dataclasses import dataclass

# No commas allowed: principals are passed to Postgres as a comma-joined string.
PRINCIPAL_RE = re.compile(r"^(user|role|group):[A-Za-z0-9_.@-]{1,64}$")


def validate_principal(p: str) -> str:
    if not isinstance(p, str) or not PRINCIPAL_RE.fullmatch(p):
        raise ValueError(f"invalid principal: {p!r}")
    return p


@dataclass(frozen=True)
class AuthContext:
    tenant_id: str
    user_id: str
    roles: tuple[str, ...] = ()
    groups: tuple[str, ...] = ()

    @property
    def principals(self) -> list[str]:
        raw = [f"user:{self.user_id}"]
        raw += [f"role:{r}" for r in self.roles]
        raw += [f"group:{g}" for g in self.groups]
        return sorted({validate_principal(p) for p in raw})

    @property
    def principals_csv(self) -> str:
        return ",".join(self.principals)
