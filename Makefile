.PHONY: up down reset logs migrate seed seed-docker api worker test security benchmark evaluate diagnosis clean

up:          ; docker compose up -d --build
down:        ; docker compose down
reset:       ; docker compose down -v
logs:        ; docker compose logs -f api worker
migrate:     ; alembic upgrade head
seed:        ; python -m scripts.seed
seed-docker: ; docker compose run --rm api python -m scripts.seed
api:         ; uvicorn app.main:app --reload
worker:      ; python -m app.worker
test:        ; python -m pytest
security:    ; python -m pytest tests/security tests/test_reranker_security.py -v
benchmark:   ; python -m tests.test_week6_evaluation
evaluate:    ; python -m tests.test_week5_quality
diagnosis:   ; python -m tests.test_citation_diagnosis
clean:       ; rm -rf .pytest_cache `find . -name __pycache__`
