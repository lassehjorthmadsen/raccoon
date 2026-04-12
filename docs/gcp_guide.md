# GCP Training Guide for Raccoon

Quick reference for running training on Google Cloud Platform.

## Key Concepts

**VM (Virtual Machine)**: A computer in the cloud that we rent from Google. Ours has a GPU (T4) for faster neural network training. We connect to it over the internet and use it like a remote terminal.

**Spot VM**: A cheaper VM (~70% discount) that Google can shut down ("preempt") at any time if they need the hardware for other customers. Our training saves checkpoints regularly, so we just restart and resume if this happens.

**SSH (Secure Shell)**: A way to get a terminal session on the remote VM. `gcloud compute ssh` handles all the connection details (IP address, keys) automatically.

**tmux (Terminal Multiplexer)**: A program that keeps terminal sessions alive on the VM even when you disconnect. Without tmux, closing your SSH connection would kill the training process. With tmux, training runs independently of your connection — you can close your laptop, go to sleep, and reconnect later to check progress.

**GCS (Google Cloud Storage)**: Google's file storage service, like a shared drive in the cloud. We use a "bucket" (`gs://raccoon-training-lhm`) to back up checkpoints and logs so they survive even if we delete the VM.

**gsutil**: Command-line tool for moving files to/from GCS. Think of it like `cp` but for cloud storage. `gsutil -m rsync` syncs a local directory to a cloud bucket (the `-m` flag means "multi-threaded" for speed).

**venv (Virtual Environment)**: An isolated Python installation inside `~/raccoon/.venv/`. It needs to be activated (`source .venv/bin/activate`) every time you open a new terminal on the VM so that `python` finds the right packages.

## Your Setup

| Resource | Value |
|----------|-------|
| GCP Project | `raccoon-training-493009` |
| VM Name | `raccoon-gpu` |
| Zone | `us-central1-a` |
| Machine | n1-standard-4 + T4 GPU (spot) |
| GCS Bucket | `gs://raccoon-training-lhm` |
| Repo on VM | `~/raccoon` |

## Daily Workflow

### 1. Start the VM

The VM is stopped between training runs to save money. Start it when you're ready to train.

```bash
gcloud compute instances start raccoon-gpu --zone=us-central1-a
```

### 2. SSH into the VM

Opens a terminal session on the remote VM. You'll see the prompt change to `lasse@raccoon-gpu`.

```bash
gcloud compute ssh raccoon-gpu --zone=us-central1-a
```

### 3. Activate the environment (every SSH session)

The venv isn't activated automatically. You need to do this every time you SSH in.

```bash
cd ~/raccoon
source .venv/bin/activate
```

### 4. Pull latest code (if you pushed changes from your iMac)

If you edited code locally and ran `git push`, pull those changes onto the VM.

```bash
git pull
pip install -e ".[dev,gnubg]"   # only needed if dependencies changed
```

### 5. Start training in tmux

Start a tmux session first, then run training inside it. This way training survives SSH disconnects.

```bash
tmux new -s train
```

Then run your training command (see Training Commands below).

### 6. Detach tmux (training keeps running)

This disconnects you from the tmux session without stopping it. Training continues in the background on the VM.

Press `Ctrl-B`, release, then press `D`.

You can then safely close the SSH connection (`exit`) and even stop your iMac.

### 7. Reconnect to tmux later

SSH back into the VM, then reattach to the running tmux session to see training output.

```bash
tmux attach -t train
```

### 8. Stop the VM when done

From your local iMac (or any terminal with gcloud). This stops compute billing but keeps your disk (checkpoints, logs) intact.

```bash
gcloud compute instances stop raccoon-gpu --zone=us-central1-a
```

## Training Commands

### Fresh training run

```bash
python scripts/train.py \
  --experiment-name NAME \
  --channels 128 --num-blocks 6 \
  --iterations 500 --games-per-iter 50 --simulations 200 \
  --training-steps 100 --batch-size 256 \
  --replay-size 500000 --checkpoint-every 10
```

### Resume from checkpoint

```bash
python scripts/train.py \
  --experiment-name NAME \
  --iterations REMAINING \
  --games-per-iter 50 --simulations 200 \
  --training-steps 100 --batch-size 256 \
  --replay-size 500000 --checkpoint-every 10 \
  --resume checkpoints/iter_XXXX.pt
```

Note: `--channels` and `--num-blocks` are read from the checkpoint on resume.

### Key training parameters

| Flag | What it does | Default |
|------|-------------|---------|
| `--experiment-name` | Label for this run (logged in JSONL) | empty |
| `--channels` | Network width | 128 |
| `--num-blocks` | Network depth (residual blocks) | 6 |
| `--iterations` | Number of train cycles to run | 100 |
| `--games-per-iter` | Self-play games per cycle | 50 |
| `--simulations` | MCTS simulations per move | 100 |
| `--training-steps` | SGD steps per cycle | 100 |
| `--batch-size` | Positions per SGD step | 256 |
| `--replay-size` | Max positions in replay buffer | 100,000 |
| `--checkpoint-every` | Save checkpoint every N iterations | 10 |
| `--resume` | Path to checkpoint to resume from | none |
| `--lr` | Learning rate | 0.001 |
| `--weight-decay` | L2 regularization | 0.0001 |

## Evaluate Against GNUBG

Run Raccoon vs GNUBG directly on the VM. The `gnubg-nn` package provides GNUBG's neural network as a Python library, so no separate GNUBG installation is needed. You can also download a checkpoint and evaluate locally on your iMac.

```bash
python scripts/eval_gnubg.py --checkpoint checkpoints/iter_0500.pt --games 100 --simulations 200
```

## Evaluate Against Earlier Checkpoint

Pit two checkpoints against each other to measure improvement over training. This runs cubeless money games between them, alternating who goes first, and reports win rate and equity (points per game).

### Latest checkpoint vs an earlier one

```bash
python scripts/evaluate.py --checkpoint1 checkpoints/iter_0490.pt --checkpoint2 checkpoints/iter_0200.pt --games 100 --simulations 50
```

### Latest checkpoint vs a random (untrained) network

Useful as a sanity check — a trained network should crush a random one. The `--random` flag is the default, so you can omit it.

```bash
python scripts/evaluate.py --checkpoint1 checkpoints/iter_0490.pt --games 50 --simulations 50
```

### What the output means

- **Win rate**: percentage of games won by player 1 (the first checkpoint)
- **Equity (ppg)**: average points per game. Positive means player 1 is stronger. Values above +0.1 ppg indicate a clear advantage; above +0.5 is a large gap.
- **Gammons/backgammons**: wins worth 2 or 3 points. A better player wins more gammons and loses fewer.

Results are logged to `logs/eval_log.jsonl` for later analysis.

## Sync Results to GCS

GCS backup protects against preemption and lets you download results to your iMac. Replace `EXPNAME` with your experiment name (e.g., `exp001-6x128-200sims`).

### Manual one-time sync

`rsync` copies only new/changed files. `-r` means recursive (include subdirectories). `-m` means multi-threaded (faster).

```bash
gsutil -m rsync -r checkpoints/ gs://raccoon-training-lhm/experiments/EXPNAME/checkpoints/
gsutil -m rsync -r logs/ gs://raccoon-training-lhm/experiments/EXPNAME/logs/
```

### Auto-sync loop

Runs in a second tmux pane (create one with `Ctrl-B`, then `C`). Syncs every 5 minutes automatically. The `while true` loop runs forever until you stop it with `Ctrl-C`.

```bash
while true; do
  gsutil -m rsync -r checkpoints/ gs://raccoon-training-lhm/experiments/EXPNAME/checkpoints/
  gsutil -m rsync -r logs/ gs://raccoon-training-lhm/experiments/EXPNAME/logs/
  sleep 300
done
```

### Download results to your iMac

Run this on your iMac (not the VM). Downloads the experiment folder from GCS to your local project.

```bash
gsutil -m cp -r gs://raccoon-training-lhm/experiments/EXPNAME/ ./experiments/
```

## Check Training Progress

### From the VM

`tail -5` shows the last 5 lines of the log file. The Python one-liner parses the JSON and prints a readable summary.

```bash
tail -5 logs/training_log.jsonl | python3 -c "
import sys, json
for l in sys.stdin:
    d = json.loads(l)
    if 'iteration' in d:
        print(f'Iter {d[\"iteration\"]}: p_loss={d[\"policy_loss\"]}, v_loss={d[\"value_loss\"]}, time={d[\"total_time\"]}s')
"
```

### From your iMac (via GCS, if syncing)

Reads the log file directly from cloud storage without needing to SSH into the VM.

```bash
gsutil cat gs://raccoon-training-lhm/experiments/EXPNAME/logs/training_log.jsonl | tail -5
```

### Check which checkpoints exist

```bash
ls checkpoints/
```

## tmux Cheat Sheet

tmux uses a two-step keyboard shortcut: press `Ctrl-B` first (the "prefix"), release it, then press the command key. Think of `Ctrl-B` as saying "hey tmux, the next key is for you, not the program running inside."

| Action | Keys | What it does |
|--------|------|-------------|
| New session | `tmux new -s NAME` | Creates a named session (run as a regular command) |
| Detach | `Ctrl-B`, then `D` | Disconnects from session; it keeps running |
| Reattach | `tmux attach -t NAME` | Reconnects to a running session |
| New window | `Ctrl-B`, then `C` | Opens a second terminal inside tmux (for GCS sync) |
| Next window | `Ctrl-B`, then `N` | Switches between windows |
| List sessions | `tmux ls` | Shows all running tmux sessions |
| Kill session | `tmux kill-session -t NAME` | Stops a session and everything in it |

## VM Management

All commands run from your iMac. The `--zone` flag tells GCP where the VM lives.

```bash
# Check if the VM is running or stopped
gcloud compute instances describe raccoon-gpu --zone=us-central1-a --format="value(status)"

# Start the VM (takes ~30 seconds)
gcloud compute instances start raccoon-gpu --zone=us-central1-a

# Stop the VM (keeps disk with all files, stops hourly billing)
gcloud compute instances stop raccoon-gpu --zone=us-central1-a

# Delete the VM entirely (destroys disk and all files — only do this if everything is backed up to GCS)
gcloud compute instances delete raccoon-gpu --zone=us-central1-a
```

## Costs

We pay for three things: compute time (VM running), disk storage (even when stopped), and cloud storage (GCS bucket). The VM is by far the biggest cost — everything else is negligible.

| Resource | Rate | When charged |
|----------|------|-------------|
| Spot VM (running) | ~$0.15/hr | Only while the VM is running |
| Stopped VM disk | ~$8/month | Continuously, even when VM is stopped |
| GCS storage | ~$0.02/GB/month | For stored checkpoints/logs |

**Key rule**: Always stop the VM when not training. A forgotten running VM costs $3.60/day.

## Handling Preemption

Since we use a spot VM, Google can reclaim it at any time (typically after a few hours). This is the tradeoff for the 70% discount. When it happens:

1. VM status becomes `TERMINATED` (check with `gcloud compute instances describe`)
2. Training stops mid-iteration. The tmux session is lost.
3. **But**: the disk survives. All checkpoints and logs are still there.
4. Just restart the VM, SSH in, activate venv, and resume from the last saved checkpoint.

To calculate `--iterations` for resume: total desired minus the checkpoint iteration minus 1.
Example: want 500 total, last checkpoint is iter_0080 → `--iterations 419 --resume checkpoints/iter_0080.pt` (runs 81..499).

**Tip**: Preemption is annoying but not harmful. We lose at most `--checkpoint-every` iterations of work (default 10, i.e., ~15 minutes). Setting up GCS sync means even the disk data has a backup.

## Experiment Naming Convention

Each training run gets a unique name so we can compare results. The name encodes the key parameters at a glance.

Format: `expNNN-{blocks}x{channels}-{sims}sims[-tag]`

- `expNNN` — sequential experiment number
- `{blocks}x{channels}` — network architecture (e.g., 6x128 = 6 residual blocks, 128 channels)
- `{sims}sims` — MCTS simulations per move
- `[-tag]` — optional extra info

Examples:
- `exp000-6x128-25sims-validation` — first test run, small params
- `exp001-6x128-200sims` — same network, more search
- `exp002-10x128-200sims` — bigger network
- `exp003-10x256-200sims-5output` — if we switch to 5-output value head

## Planned Training Phases

We scale up gradually. Each phase increases one or two parameters. We evaluate against GNUBG between phases to decide whether to continue or change direction.

| Phase | Network | Sims | Games/iter | Iters | Est. cost | Goal |
|-------|---------|------|------------|-------|-----------|------|
| 0 (current) | 6x128 | 25 | 10 | 500 | ~$2 | Validate VM pipeline works |
| 1 | 6x128 | 200 | 50 | 500 | ~$0.50 | Better training signal (more search) |
| 2 | 10x128 | 200 | 100 | 1000 | ~$3-4 | More network capacity |
| 3 | 10x256 | 200-400 | 200 | 2000 | ~$12-18 | Serious scaling |
| 4 | 20x256 | 400 | 500 | 5000 | ~$75+ | Full power (if needed) |
