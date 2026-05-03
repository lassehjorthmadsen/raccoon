.PHONY: setup test smoke train eval eval-gnubg play

PYTHON := .venv/bin/python3

setup:
	python3 -m venv .venv
	$(PYTHON) -m pip install -e ".[dev]"

test:
	$(PYTHON) -m pytest tests/ -v

smoke:
	$(PYTHON) scripts/train.py --experiment-name smoke --iterations 2 --games-per-iter 3 --simulations 10

train:
	@if [ -z "$(NAME)" ]; then echo "Usage: make train NAME=expNNN-6x128-200sims"; exit 1; fi
	$(PYTHON) scripts/train.py --experiment-name $(NAME)

eval:
	$(PYTHON) scripts/evaluate.py

eval-gnubg:
	$(PYTHON) scripts/eval_gnubg.py

play:
	$(PYTHON) scripts/play.py
