PYTHON ?= python

.PHONY: run train test

run:
	$(PYTHON) application.py

train:
	$(PYTHON) -m src.train

test:
	$(PYTHON) -m pytest
