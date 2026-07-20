PYTHON = uv run python3
MAIN = -m src
SRC = src/

all: install

uv.lock: pyproject.toml Makefile
	@echo "Installing dependencies using uv..."
	uv sync
	@touch uv.lock

install: uv.lock
	uv sync

run: install
	@echo "Running the program..."
	$(PYTHON) $(MAIN) $(ARGS) 

debug: install
	@echo "Starting debug mode..."
	$(PYTHON) -m pdb $(MAIN) $(ARGS)

lint:
	@echo "Running standard linting..."
	uv run flake8 $(SRC)
	uv run mypy $(SRC)

lint-strict:
	@echo "Running strict linting..."
	uv run flake8 $(SRC)
	uv run mypy --strict $(SRC)

profile: install
	@echo "Profiling the application..."
	$(PYTHON) -m cProfile -s cumtime src/__main__.py > profil_complet.txt

clean:
	@echo "Cleaning up..."
	rm -rf .mypy_cache \
	       .pytest_cache \
	       .ruff_cache \
	       profil_complet.txt \
	       data/output
	find . -type d -name "__pycache__" -exec rm -rf {} +

.PHONY: all install run debug lint lint-strict clean profile