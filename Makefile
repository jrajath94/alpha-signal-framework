install:
	pip install -e .
test:
	pytest tests/ -v
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
