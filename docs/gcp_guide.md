# GCP Training Guide for Raccoon

Quick reference for running training on Google Cloud Platform.

## Key Concepts

**VM (Virtual Machine)**: A computer in the cloud that we rent from Google. Ours has a GPU (T4) for faster neural network training. We connect to it over the internet and use it like a remote terminal.

**Spot VM**: A cheaper VM (~70% discount) that Google can shut down ("preempt") at any time if they need the hardware for other customers. Our training saves checkpoints regularly, so we just restart and resume if this happens.

**SSH (Secure Shell)**: A way to get a terminal session on the remote VM. `gcloud compute ssh` handles all the connection details (IP address, keys) automatically.

**tmux (Terminal Multiplexer)**: A program that keeps terminal sessions alive on the VM even when you disconnect. Without tmux, closing your SSH connection would kill the training process. With tmux, training runs independently of your connection — you can close your laptop, go to sleep, and reconnect later to check progress.

**GCS (Google Cloud Storage)**: Google's file storage service, like a shared drive in the cloud. We use a "bucket" (`gs://raccoon-training-lhm`) to back up checkpoints and logs so they survive even if we delete the VM.

**gcloud storage**: Command-line tool for moving files to/from GCS. Think of it like `cp` but for cloud storage. `gcloud storage rsync` syncs a local directory to a cloud bucket. (Note: the older `gsutil` tool exists but has auth issues on our VM — always use `gcloud storage` instead.)

**venv (Virtual Environment)**: An isolated Python installation inside `~/raccoon/.venv/`. It needs to be activated (`source .venv/bin/activate`) every time you open a new terminal on the VM so that `python` finds the right packages.

## Your Setup

| Resource | Value |
|----------|-------|
| GCP Project | `raccoon-training-493009` |
| VM Name | `raccoon-gpu` |
| Zone | `europe-west1-b` |
| Machine | n1-standard-4 + T4 GPU (spot) |
| GCS Bucket | `gs://raccoon-training-lhm` |
| Repo on VM | `~/raccoon` |

## Daily Workflow

### 1. Start the VM

The VM is stopped between training runs to save money. Start it when you're ready to train.

```bash
gcloud compute instances start raccoon-gpu --zone=europe-west1-b
```

### 2. SSH into the VM

Opens a terminal session on the remote VM. You'll see the prompt change to `lasse@raccoon-gpu`.

```bash
gcloud compute ssh raccoon-gpu --zone=europe-west1-b
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
gcloud compute instances stop raccoon-gpu --zone=europe-west1-b
```

## Training Commands

`--experiment-name` is required. Outputs go to `experiments/<name>/{checkpoints,logs}/`.

### Fresh training run

```bash
python scripts/train.py \
  --experiment-name NAME \
  --channels 128 --num-blocks 6 \
  --iterations 500 --games-per-iter 50 --simulations 200 \
  --training-steps 100 --batch-size 256 \
  --replay-size 500000 --checkpoint-every 1
```

### Resume from checkpoint

```bash
python scripts/train.py \
  --experiment-name NAME \
  --iterations REMAINING \
  --games-per-iter 50 --simulations 200 \
  --training-steps 100 --batch-size 256 \
  --replay-size 500000 --checkpoint-every 1 \
  --resume experiments/NAME/checkpoints/iter_XXXX.pt
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
| `--checkpoint-every` | Save checkpoint every N iterations (use 1 on spot VMs — each checkpoint is only 22MB) | 10 |
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

### File structure

Training writes directly under `experiments/EXPNAME/`. The same layout is used on the VM, in GCS, and on your iMac — so sync is a straight mirror.

```
experiments/exp001-6x128-200sims/checkpoints/iter_0000.pt
experiments/exp001-6x128-200sims/checkpoints/iter_0001.pt
experiments/exp001-6x128-200sims/logs/training_log.jsonl
```

### Manual one-time sync

`rsync` copies only new/changed files. `--recursive` includes subdirectories.

```bash
gcloud storage rsync experiments/EXPNAME/ gs://raccoon-training-lhm/experiments/EXPNAME/ --recursive
```

### Auto-sync loop

Runs in a second tmux pane (create one with `Ctrl-B`, then `C`). Syncs every 5 minutes automatically. The `while true` loop runs forever until you stop it with `Ctrl-C`.

```bash
while true; do
  gcloud storage rsync experiments/EXPNAME/ gs://raccoon-training-lhm/experiments/EXPNAME/ --recursive
  echo "Synced at $(date)"
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
tail -5 experiments/EXPNAME/logs/training_log.jsonl | python3 -c "
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
gcloud storage cat gs://raccoon-training-lhm/experiments/EXPNAME/logs/training_log.jsonl | tail -5
```

### Check which checkpoints exist

```bash
ls experiments/EXPNAME/checkpoints/
```

### Quick status check from local machine

Check current iteration and recent metrics without SSH:

```bash
gcloud compute ssh raccoon-gpu --zone=us-central1-a --command="sudo tail -3 /home/lasse/raccoon/logs/training_log.jsonl | grep -o 'iteration\": [0-9]*'"
```

Detailed status with losses:

```bash
gcloud compute ssh raccoon-gpu --zone=us-central1-a --command="echo 'Current time:' && date -u && echo '---' && sudo tail -1 /home/lasse/raccoon/logs/training_log.jsonl | python3 -c \"import sys,json; d=json.loads(sys.stdin.read()); print(f'Iteration: {d[\\\"iteration\\\"]}'); print(f'Policy loss: {d[\\\"policy_loss\\\"]:.4f}'); print(f'Value loss: {d[\\\"value_loss\\\"]:.4f}'); print(f'Time/iter: {d[\\\"total_time\\\"]:.1f}s')\""
```

Check self-play progress:

```bash
gcloud compute ssh raccoon-gpu --zone=us-central1-a --command="sudo tail -5 /home/lasse/raccoon/logs/train_exp001.log"
```

## Stopping Training

### Stop at a specific iteration

When training reaches your target iteration, stop it to save costs.

**Option 1: Kill the tmux session (recommended)**

```bash
# SSH into the VM
gcloud compute ssh raccoon-gpu --zone=us-central1-a

# Switch to user lasse and kill the training session
sudo -u lasse tmux kill-session -t train

# Verify it stopped
sudo -u lasse tmux ls

# Exit SSH
exit
```

**Option 2: Stop the entire VM**

```bash
# From your local machine - stops billing immediately
gcloud compute instances stop raccoon-gpu --zone=us-central1-a
```

### Stopping in the middle of an iteration

Training saves checkpoints every 10 iterations by default. If you stop mid-iteration, you'll lose progress since the last checkpoint (at most ~45 minutes of work). Best practice: wait for the next checkpoint to be saved before stopping.

## Downloading Results

### Download specific checkpoints to local machine

```bash
# Create local directory
mkdir -p phase1_checkpoints phase1_logs

# Download a specific checkpoint
gcloud compute scp --zone=us-central1-a raccoon-gpu:/home/lasse/raccoon/checkpoints/iter_0150.pt ./phase1_checkpoints/

# Download training log
gcloud compute scp --zone=us-central1-a raccoon-gpu:/home/lasse/raccoon/logs/training_log.jsonl ./phase1_logs/

# Download evaluation results
gcloud compute scp --zone=us-central1-a raccoon-gpu:/home/lasse/raccoon/logs/eval_log.jsonl ./phase1_logs/
gcloud compute scp --zone=us-central1-a raccoon-gpu:/home/lasse/raccoon/logs/gnubg_eval_log.jsonl ./phase1_logs/
```

### Download multiple checkpoints

```bash
# Download checkpoints at key intervals (0, 50, 100, 150)
for i in 0000 0050 0100 0150; do
  gcloud compute scp --zone=us-central1-a \
    raccoon-gpu:/home/lasse/raccoon/checkpoints/iter_${i}.pt \
    ./phase1_checkpoints/
done
```

### Handle permission issues when downloading

If you get "permission denied" errors, copy files to /tmp first:

```bash
# On VM: Copy to accessible location
gcloud compute ssh raccoon-gpu --zone=us-central1-a --command="sudo cp /home/lasse/raccoon/checkpoints/iter_0150.pt /tmp/ && sudo chmod 644 /tmp/iter_0150.pt"

# Then download from /tmp
gcloud compute scp --zone=us-central1-a raccoon-gpu:/tmp/iter_0150.pt ./phase1_checkpoints/
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
gcloud compute instances describe raccoon-gpu --zone=europe-west1-b --format="value(status)"

# Start the VM (takes ~30 seconds)
gcloud compute instances start raccoon-gpu --zone=europe-west1-b

# Stop the VM (keeps disk with all files, stops hourly billing)
gcloud compute instances stop raccoon-gpu --zone=europe-west1-b

# Delete the VM entirely (destroys disk and all files — only do this if everything is backed up to GCS)
gcloud compute instances delete raccoon-gpu --zone=europe-west1-b
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
Example: want 500 total, last checkpoint is iter_0080 → `--iterations 419 --resume experiments/EXPNAME/checkpoints/iter_0080.pt` (runs 81..499).

**Tip**: Preemption is annoying but not harmful. We lose at most `--checkpoint-every` iterations of work. Use `--checkpoint-every 1` on spot VMs — each checkpoint is only 22MB so the cost is negligible. Always run the GCS auto-sync loop so even the disk data has a backup.

**If preemptions are frequent**: The zone may have high spot demand. Move to a different zone by snapshotting the disk and recreating the VM (we moved from `us-central1-a` to `europe-west1-b` for this reason). T4 GPUs are available in many zones — run `gcloud compute accelerator-types list --filter="name=nvidia-tesla-t4"` to see options.

## Auto-Recovery Watchdog

Spot preemptions happen at unpredictable times — sometimes multiple per day. Manually restarting each time is tedious, so we have a two-part watchdog that handles it automatically.

- **`scripts/watch_vm.sh`** runs locally on the iMac. Polls VM status every 5 min; on `TERMINATED`, starts the VM (retrying on stockouts) and SSHes in to relaunch training.
- **`scripts/resume_training.sh`** lives on the VM (committed in the repo). Finds the latest checkpoint in `experiments/<name>/checkpoints/`, launches training in a `train` tmux session, and kicks off a GCS sync loop in a `sync` tmux session. Idempotent — does nothing if `train` is already alive.

### Start the watchdog

Wrap in `systemd-inhibit` so the iMac won't suspend and freeze the watchdog:

```bash
nohup systemd-inhibit --what=sleep:idle --who=watch_vm --why="keep VM watchdog alive" \
  ./scripts/watch_vm.sh EXPNAME ITERATIONS > /tmp/watch_vm.log 2>&1 &
disown
```

### Verify

```bash
pgrep -af watch_vm.sh                    # watchdog process alive
systemd-inhibit --list | grep watch_vm   # sleep inhibit registered
tail -f /tmp/watch_vm.log                # activity log
```

### Stop the watchdog

```bash
pkill -f watch_vm.sh
```

### Tuning training params

Training flags are hardcoded in `scripts/resume_training.sh`. Edit that file (and `git push` / `git pull` on the VM) when changing params for a new experiment.

### Caveats

- Watchdog pauses if the iMac suspends; `systemd-inhibit` prevents idle/explicit sleep but **not** lid-close sleep on a laptop. Our iMac is a desktop, so this is fine.
- If the zone is fully out of T4 capacity (persistent stockout), the watchdog will keep retrying every 2 min indefinitely. Check `/tmp/watch_vm.log` if the VM seems stuck down.

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

**Note**: Actual Phase 1 timing is ~44 min/iteration (not the initial estimate), so costs are updated based on real measurements.

| Phase | Network | Sims | Games/iter | Iters | Time/iter | Est. cost | Goal |
|-------|---------|------|------------|-------|-----------|-----------|------|
| 0 (completed) | 6x128 | 25 | 10 | 499 | ~70s | ~$2 | Validate VM pipeline works |
| 1 (current) | 6x128 | 200 | 50 | 150 | ~44min | ~$17 | Better training signal (more search) |
| 1-full | 6x128 | 200 | 50 | 500 | ~44min | ~$55 | Full Phase 1 if needed |
| 2 | 10x128 | 200 | 100 | 500-1000 | TBD | ~$50-100 | More network capacity |
| 3 | 10x256 | 200-400 | 200 | 1000-2000 | TBD | ~$100-200 | Serious scaling |

**Phase 0 Results:**
- Validated training loop works
- Internal eval: iter_498 beats iter_290 by +0.06 ppg
- GNUBG 0-ply: 0-10 loss, -2.5 ppg (too weak)
- Conclusion: Need stronger training signal

**Phase 1 Plan:**
- Target: 150 iterations (~4.6 days, ~$17)
- Evaluate vs GNUBG 0-ply at completion
- Decide whether to continue or scale up network
