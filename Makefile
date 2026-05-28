.PHONY: setup test smoke train eval eval-gnubg play download-wildbg pretrain-smoke pretrain

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

download-wildbg:
	./scripts/download_wildbg.sh

pretrain-smoke:
	$(PYTHON) scripts/pretrain.py --experiment-name pretrain-smoke --epochs 1 --max-positions 2000

pretrain:
	@if [ -z "$(NAME)" ]; then echo "Usage: make pretrain NAME=pretrain-wildbg-v1"; exit 1; fi
	$(PYTHON) scripts/pretrain.py --experiment-name $(NAME)
