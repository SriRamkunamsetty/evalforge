.PHONY: run worker test lint sast docker-up docker-down screenshots helm-lint k8s-render

run:
	API_KEY=dev-key DATA_DIR=./data flask --app "evalforge.server:create_app" run --port 8083

worker:
	API_KEY=dev-key DATA_DIR=./data python -m evalforge.worker

test:
	python -m pytest --cov=evalforge --cov-report=term-missing --cov-fail-under=80

lint:
	ruff check .

sast:
	bandit -r evalforge -ll -ii

docker-up:
	docker compose up --build -d

docker-down:
	docker compose down -v

screenshots:
	python scripts/capture_screenshots.py

helm-lint:
	helm lint deploy/helm/evalforge --strict

k8s-render:
	kubectl kustomize deploy/k8s/overlays/production
