PROJECT_DIR := $(shell pwd -P)
CACHE_DIR := $(PROJECT_DIR)/.cache/uv
TMP_DIR := $(PROJECT_DIR)/.tmp
HF_CACHE_DIR ?= /tmp/call-me-maybe-hf

export UV_CACHE_DIR := $(CACHE_DIR)
export TMPDIR := $(TMP_DIR)
export HF_HOME := $(HF_CACHE_DIR)

PYTHON = uv run python3
MAIN = -m src
SRC = src/

all: build

uv.lock: pyproject.toml Makefile
	@echo "Installing dependencies using uv..."
	@mkdir -p "$(CACHE_DIR)" "$(TMP_DIR)"
	uv sync
	@touch uv.lock

install: uv.lock
	@mkdir -p "$(CACHE_DIR)" "$(TMP_DIR)" "$(HF_CACHE_DIR)"
	uv sync

build:
	@echo "Building distribution packages..."
	@mkdir -p "$(CACHE_DIR)" "$(TMP_DIR)" "$(HF_CACHE_DIR)"
	uv build

run: install
	@echo "Running the program..."
	$(PYTHON) $(MAIN) $(ARGS) 

debug: install
	@echo "Starting debug mode..."
	$(PYTHON) -m pdb $(MAIN) $(ARGS)

test: install
	@echo "Running unit tests..."
	$(PYTHON) -m unittest discover -s tests -v

lint:
	@echo "Running standard linting..."
	uv run flake8 .
	uv run mypy . --warn-return-any --warn-unused-ignores \
		--ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

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

.PHONY: all install build run debug test lint lint-strict clean profile
