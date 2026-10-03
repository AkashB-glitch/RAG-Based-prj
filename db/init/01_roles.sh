#!/bin/bash
# Runs once, on first DB start. Creates the two least-privilege login roles.
# Neither role is superuser, owns tables, or has BYPASSRLS -> RLS always applies to them.
set -euo pipefail
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" \
     -v app_pw="$APP_DB_PASSWORD" -v worker_pw="$WORKER_DB_PASSWORD" <<-'EOSQL'
  CREATE ROLE app_user   LOGIN PASSWORD :'app_pw'    NOSUPERUSER NOBYPASSRLS NOCREATEDB NOCREATEROLE;
  CREATE ROLE rag_worker LOGIN PASSWORD :'worker_pw' NOSUPERUSER NOBYPASSRLS NOCREATEDB NOCREATEROLE;
EOSQL
