"""Seed two tenants with deliberately overlapping topics (both have a 'salary bands' doc).
That overlap is what makes the leakage tests meaningful: a similarity search in tenant A for
'salary bands' would happily return tenant B's document if isolation were broken.

Runs as the OWNER role. Idempotent. Documents are queued for the worker to ingest.
"""
import hashlib
from pathlib import Path

import psycopg

from app.config import settings

DOCS = Path(__file__).resolve().parent.parent / "seed" / "docs"

TENANTS = {
    "acme": {
        "name": "Acme Corp",
        "users": [
            ("alice@acme.test", ["employee", "hr"], []),
            ("bob@acme.test", ["employee"], ["engineering"]),
            ("erin@acme.test", ["employee"], []),
        ],
        "docs": [
            ("Acme 2026 salary bands", "acme_salaries_2026.md", ["role:hr"]),
            ("Payments on-call runbook", "acme_oncall_runbook.md", ["group:engineering"]),
            ("Employee handbook", "acme_handbook.md", ["role:employee"]),
        ],
    },
    "globex": {
        "name": "Globex Ltd",
        "users": [("carol@globex.test", ["employee", "hr"], [])],
        "docs": [("Globex 2026 salary bands", "globex_salaries_2026.md", ["role:hr"])],
    },
}


def owner_dsn() -> str:
    return settings.owner_database_url.replace("postgresql+psycopg://", "postgresql://")


def main() -> None:
    with psycopg.connect(owner_dsn()) as conn, conn.cursor() as cur:
        for slug, t in TENANTS.items():
            cur.execute(
                "INSERT INTO tenants (slug, name) VALUES (%s, %s) "
                "ON CONFLICT (slug) DO UPDATE SET name = EXCLUDED.name RETURNING id",
                (slug, t["name"]),
            )
            tenant_id = cur.fetchone()[0]
            user_ids = {}
            for email, roles, groups in t["users"]:
                cur.execute(
                    "INSERT INTO users (tenant_id, email, roles, groups) VALUES (%s, %s, %s, %s) "
                    "ON CONFLICT (tenant_id, email) DO UPDATE SET roles = EXCLUDED.roles, groups = EXCLUDED.groups "
                    "RETURNING id",
                    (tenant_id, email, roles, groups),
                )
                user_ids[email] = cur.fetchone()[0]
            uploader = next(iter(user_ids.values()))
            for title, filename, acl in t["docs"]:
                cur.execute("SELECT 1 FROM documents WHERE tenant_id = %s AND title = %s", (tenant_id, title))
                if cur.fetchone():
                    continue
                raw = (DOCS / filename).read_bytes()
                cur.execute(
                    "INSERT INTO documents (tenant_id, title, mime_type, raw, content_sha256, acl, uploaded_by) "
                    "VALUES (%s, %s, 'text/markdown', %s, %s, %s, %s) RETURNING id",
                    (tenant_id, title, raw, hashlib.sha256(raw).hexdigest(), acl, uploader),
                )
                doc_id = cur.fetchone()[0]
                cur.execute("INSERT INTO ingest_jobs (tenant_id, document_id) VALUES (%s, %s)", (tenant_id, doc_id))
            print(f"seeded tenant {slug}: users={list(user_ids)}")
    print("done. start the worker to ingest queued documents.")


if __name__ == "__main__":
    main()
