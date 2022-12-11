VENV := .venv

ifeq ($(OS),Windows_NT)
   BIN=$(VENV)/Scripts
else
   BIN=$(VENV)/bin
endif

export PATH := $(BIN):$(PATH)

PROJECT := service

# Prepare

.venv:
	poetry install --no-root
	poetry check

setup: .venv


# Clean

clean:
	rm -rf .pytest_cache
	rm -rf $(VENV)


# Format

isort_fix: .venv
	isort $(PROJECT)

format: isort_fix


# Lint

flake8: .venv
	flake8 --statistics --config $(PROJECT)

black: .venv
	black --check $(PROJECT)


lint: black flake8


# All

all: setup format lint

.DEFAULT_GOAL = all
