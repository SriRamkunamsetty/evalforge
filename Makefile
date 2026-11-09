run:
	API_KEY=dev-key DATA_DIR=./data flask --app "evalforge.server:create_app" run --port 8083
test:
	python -m pytest --cov=evalforge --cov-report=term-missing
lint:
	ruff check .
docker:
	docker compose up --build
screenshots:
	python scripts/capture_screenshots.py
