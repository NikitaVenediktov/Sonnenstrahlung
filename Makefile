VENV := .venv

ifeq ($(OS),Windows_NT)
   BIN=$(VENV)/Scripts
else
   BIN=$(VENV)/bin
endif

export PATH := $(BIN):$(PATH)

PROJECT := sonnenstrahlung

IMAGE_NAME := sonnenstrahlung
CONTAINER_NAME := sonnenstrahlung

# Prepare

.venv:
	poetry install --no-root
	poetry check

setup: .venv


# Clean

clean:
	rm -rf $(VENV)

# Train

train: setup
	python train.py

# Run locally

local: setup
	python autopredict.py

# Docker

build:
	docker build . -t $(IMAGE_NAME)

run: build
	docker run -dp 7070:7070 --name $(CONTAINER_NAME) $(IMAGE_NAME)

# All

all: setup local

.DEFAULT_GOAL = all
