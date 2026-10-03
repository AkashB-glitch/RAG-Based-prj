# Acme Payments — On-call Runbook (Engineering)

If the payments API p95 latency exceeds 800 ms, first check the connection pool saturation dashboard.
Restart the ledger-sync worker only after draining the queue with `make drain-ledger`.
Escalate to the database on-call if replication lag exceeds 30 seconds.
