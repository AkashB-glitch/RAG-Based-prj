"""Database-level isolation tests. These talk to Postgres directly as app_user / rag_worker and
deliberately OMIT the app's WHERE clause, proving RLS alone blocks leakage (defense in depth)."""
import psycopg
import pytest

from tests.conftest import APP_DSN, WORKER_DSN, as_role, requires_db

pytestmark = requires_db


def visible_contents(conn) -> list[str]:
    # No tenant / ACL filter on purpose: whatever comes back is what RLS allowed.
    return [r[0] for r in conn.execute("SELECT content FROM chunks").fetchall()]


def test_no_context_sees_nothing(world):
    with as_role(APP_DSN, None) as conn:
        assert visible_contents(conn) == []
        assert conn.execute("SELECT count(*) FROM documents").fetchone()[0] == 0


def test_tenant_without_principals_sees_nothing(world):
    with as_role(APP_DSN, world["tenants"]["a"]) as conn:
        assert visible_contents(conn) == []


def test_hr_in_tenant_a_sees_only_tenant_a_hr_and_nothing_from_b(world):
    with as_role(APP_DSN, world["tenants"]["a"], ["role:hr", "user:x"]) as conn:
        rows = visible_contents(conn)
    assert rows == ["tenant a salary bands merit pool"]
    assert not any("tenant b" in r for r in rows)


def test_employee_cannot_see_hr_or_engineering_docs(world):
    with as_role(APP_DSN, world["tenants"]["a"], ["role:employee"]) as conn:
        assert visible_contents(conn) == ["tenant a handbook leave policy"]


def test_same_role_name_in_other_tenant_does_not_cross(world):
    # role:hr exists in both tenants; the tenant check keeps them apart.
    with as_role(APP_DSN, world["tenants"]["b"], ["role:hr"]) as conn:
        assert visible_contents(conn) == ["tenant b salary bands merit pool"]


def test_vector_search_without_filters_cannot_rank_in_foreign_chunks(world):
    with as_role(APP_DSN, world["tenants"]["a"], ["role:employee", "role:hr", "group:engineering"]) as conn:
        rows = conn.execute(
            "SELECT content FROM chunks ORDER BY embedding <=> (SELECT embedding FROM chunks LIMIT 1) LIMIT 50"
        ).fetchall()
    assert rows and all(r[0].startswith("tenant a") for r in rows)


def test_app_user_cannot_write_chunks(world):
    with as_role(APP_DSN, world["tenants"]["a"], ["role:hr"]) as conn:
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            conn.execute("DELETE FROM chunks")


def test_audit_log_is_append_only(world):
    t = world["tenants"]["a"]
    with as_role(APP_DSN, t, ["role:hr"]) as conn:
        conn.execute(
            "INSERT INTO retrieval_audit (request_id, tenant_id, user_id, principals, query_sha256, chunk_ids, document_ids) "
            "VALUES (gen_random_uuid(), %s, 'u', '{role:hr}', 'h', '{}', '{}')",
            (t,),
        )
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            conn.execute("UPDATE retrieval_audit SET user_id = 'someone-else'")
    with as_role(APP_DSN, t, ["role:hr"]) as conn:
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            conn.execute("DELETE FROM retrieval_audit")


def test_cannot_write_audit_rows_for_another_tenant(world):
    with as_role(APP_DSN, world["tenants"]["a"], ["role:hr"]) as conn:
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            conn.execute(
                "INSERT INTO retrieval_audit (request_id, tenant_id, user_id, principals, query_sha256, chunk_ids, document_ids) "
                "VALUES (gen_random_uuid(), %s, 'u', '{}', 'h', '{}', '{}')",
                (world["tenants"]["b"],),
            )


def test_cannot_upload_document_into_another_tenant(world):
    with as_role(APP_DSN, world["tenants"]["a"], ["user:00000000-0000-0000-0000-000000000001"],
                 user_id="00000000-0000-0000-0000-000000000001") as conn:
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            conn.execute(
                "INSERT INTO documents (tenant_id, title, mime_type, raw, content_sha256, acl, uploaded_by) "
                "VALUES (%s, 'x', 'text/plain', 'x', 'x', '{role:hr}', '00000000-0000-0000-0000-000000000001')",
                (world["tenants"]["b"],),
            )


def test_worker_cannot_attach_chunk_to_other_tenants_document(world):
    foreign_doc = world["docs"][("b", "salary")]
    with as_role(WORKER_DSN, world["tenants"]["a"]) as conn:
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            conn.execute(
                "INSERT INTO chunks (document_id, chunk_index, content, embedding) "
                "VALUES (%s, 99, 'smuggled', array_fill(0.1, ARRAY[384])::vector)",
                (foreign_doc,),
            )


def test_chunk_acl_cannot_be_chosen_by_writer(world):
    doc = world["docs"][("a", "salary")]
    with as_role(WORKER_DSN, world["tenants"]["a"]) as conn:
        conn.execute(
            "INSERT INTO chunks (document_id, chunk_index, content, embedding, acl, tenant_id) "
            "VALUES (%s, 7, 'x', array_fill(0.1, ARRAY[384])::vector, '{role:employee}', %s) RETURNING acl",
            (doc, world["tenants"]["b"]),
        )
        acl, tenant = conn.execute(
            "SELECT acl, tenant_id::text FROM chunks WHERE document_id = %s AND chunk_index = 7", (doc,)
        ).fetchone()
    assert acl == ["role:hr"] and tenant == world["tenants"]["a"]


def test_worker_cannot_change_document_acl(world):
    with as_role(WORKER_DSN, world["tenants"]["a"]) as conn:
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            conn.execute("UPDATE documents SET acl = '{role:employee}'")
