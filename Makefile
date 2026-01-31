PYTHON ?= python

.PHONY: run train test

run:
	$(PYTHON) application.py

train:
	$(PYTHON) -m src.pipeline.training_pipeline

test:
	$(PYTHON) -m unittest discover
