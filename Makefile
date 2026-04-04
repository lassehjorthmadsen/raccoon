.PHONY: setup test smoke train eval eval-gnubg play

PYTHON := .venv/bin/python3

setup:
	python3 -m venv .venv
	$(PYTHON) -m pip install -e ".[dev]"

test:
	$(PYTHON) -m pytest tests/ -v

smoke:
	$(PYTHON) scripts/train.py --iterations 2 --games-per-iter 3 --simulations 10

train:
	$(PYTHON) scripts/train.py

eval:
	$(PYTHON) scripts/evaluate.py

eval-gnubg:
	$(PYTHON) scripts/eval_gnubg.py

play:
	$(PYTHON) scripts/play.py
