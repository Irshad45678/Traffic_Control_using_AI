# Traffic Light Control with Reinforcement Learning (SUMO + TensorFlow)

This repository implements **adaptive traffic signal control** using **deep reinforcement learning** inside the **SUMO** (Simulation of Urban MObility) microscopic traffic simulator. The agent observes lane-level traffic near each intersection, selects the next green phase, and is trained with experience replay and a neural Q-function approximator. Results can be compared against a **fixed-time / static** controller that cycles phases in order.

The codebase is suitable for coursework, research prototypes, and extension to larger networks or different reward formulations.

## Table of contents

- [Features](#features) · [Technology stack](#technology-stack) · [Requirements](#requirements) · [Installation](#installation)
- [Environment variables](#environment-variables) · [Quick start](#quick-start) · [Configuration reference](#configuration-reference)
- [Methodology overview](#methodology-overview) · [Source code layout](#source-code-layout) · [SUMO scenarios](#sumo-scenarios)
- [Model artifacts and naming](#model-artifacts-and-naming) · [Outputs and logging](#outputs-and-logging) · [Evaluation metrics](#evaluation-metrics)
- [Optional GUI (`lake.py`)](#optional-gui-lakepy) · [Troubleshooting](#troubleshooting) · [Roadmap](#roadmap-and-future-work)
- [References](#references) · [License](#license) · [Contributing](#contributing)

## Features

- **Training** of per–traffic-light neural policies with **TraCI** control of SUMO.
- **Stochastic route generation** per episode (Weibull-distributed vehicle arrivals, straight vs. turn split).
- **Experience replay** with configurable buffer size and batch training after each simulated episode.
- **Testing** mode that loads saved Keras models and runs **greedy** (argmax) control.
- **Static baseline** simulation for comparison without any learned model.
- **Plots and text exports** for rewards, delays, queues, and test-time bar charts.
- **INI-based configuration** so you can switch scenarios and hyperparameters without editing Python.

## Technology stack

| Component | Role |
|-----------|------|
| **Python** | Orchestration, configuration, ML pipeline |
| **SUMO** | Microscopic traffic simulation; network, routes, signals |
| **TraCI** (`traci`) | Python API to step SUMO and read/write traffic lights and vehicles |
| **sumolib** | Read `.net.xml`, enumerate traffic lights, connections, lanes |
| **TensorFlow / Keras** | Neural Q-network training and inference (`.h5` export) |
| **NumPy** | State tensors, replay batches, numerical utilities |
| **Matplotlib** | Saved figures (PNG) for training curves and test summaries |
| **configparser** | INI files (`train_settings.ini`, `test_settings.ini`) |

## Requirements

- **Python** 3.8+ recommended; match `requirements.txt` (TensorFlow 2.8.x is pinned).
- **SUMO** compatible with the pinned `traci` / `sumolib` versions; install from the official Eclipse SUMO distribution.
- **`SUMO_HOME`** must point to the SUMO installation root. The project prepends `$SUMO_HOME/tools` to `sys.path` so `traci` and `sumolib` match your SUMO build.
- **Working directory**: run commands from the **repository root** so `sumo_files/...` and `models/...` resolve.
- **GPU (optional)**: TensorFlow uses a GPU if available; CPU-only works but is slower for long training.

## Installation

```bash
git clone <your-repository-url>
cd <your-repository-folder>
```

Create a virtual environment and install dependencies:

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux / macOS
# python3 -m venv .venv && source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

Set **`SUMO_HOME`** (examples):

| Shell | Command |
|-------|---------|
| Windows CMD | `set SUMO_HOME=C:\Program Files (x86)\Eclipse\Sumo` |
| Windows PowerShell | `$env:SUMO_HOME = "C:\path\to\sumo"` |
| Linux / macOS | `export SUMO_HOME=/path/to/sumo` |

Ensure `sumo` / `sumo-gui` are on your `PATH` (installers often configure this when `SUMO_HOME` is set).

## Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `SUMO_HOME` | **Yes** | Root of your SUMO install; used to locate `tools/` for Python bindings. |

If `SUMO_HOME` is missing, the program exits with an error (see `src/tools.py`).

## Quick start

| Task | Command | Config file |
|------|---------|---------------|
| Train RL models | `python src/train.py` | `train_settings.ini` |
| Evaluate trained models | `python src/test.py` | `test_settings.ini` |
| Static (non-RL) baseline | `python src/static_simulation.py` | `test_settings.ini` |

After training, set `test_settings.ini` → `[dir]` → `model_folder` to the folder under `models/` that contains your `.h5` files (per traffic-light subfolders).

### Typical workflow

1. Edit **`train_settings.ini`**: choose `sumo_file`, `model_folder`, `num_cars`, `total_episodes`, and agent/memory settings.
2. Run **`python src/train.py`**. Watch `debug.log` and console output; new folders appear under `models/<model_folder>/`.
3. Copy or note the **actual** output folder name (training may append ` (1)` if the name already exists—see `set_path` in `src/tools.py`).
4. Edit **`test_settings.ini`**: set `model_folder` to that folder, matching `sumo_file` and `input_dim` to training.
5. Run **`python src/test.py`** for RL evaluation, or **`python src/static_simulation.py`** for the fixed-phase baseline.
6. Compare PNG and `test_*_data.txt` files under each traffic-light subdirectory.

### Repository tree (conceptual)

```
├── src/                      # Application code
│   ├── interface/            # TrafficLight, Route helpers
│   └── testing/              # Ad-hoc scripts (not main pipeline)
├── sumo_files/               # Networks, configs, route templates
├── models/                   # Trained weights and experiment outputs (generated)
├── train_settings.ini
├── test_settings.ini
├── requirements.txt
├── LICENSE
└── README.md
```

### Default settings (illustrative)

Your copies may differ; defaults in the repo look like:

**Training** (`train_settings.ini`): `input_dim = 16`, `total_episodes = 100`, `max_step = 5400`, `num_cars = 1000`, example `sumo_file = 1TL/1TL`, `model_folder = testRUN`.

**Testing** (`test_settings.ini`): example `max_step = 3600`, `sumo_file` pointing at a scenario such as `Roxas/All/Roxas`, and `model_folder` pointing at a trained run under `models/`.

## Configuration reference

### `train_settings.ini`

| Section | Keys | Meaning |
|---------|------|---------|
| `[agent]` | `input_dim`, `num_layers`, `batch_size`, `learning_rate` | Lane grid width, MLP depth, replay batch size, Adam learning rate |
| `[memory]` | `size_min`, `size_max` | Minimum samples before training from replay; max replay length |
| `[simulation]` | `total_episodes`, `sumo_gui`, `max_step`, `epochs`, `gamma`, `green_duration`, `yellow_duration` | Training length, headless vs GUI SUMO, steps per episode, post-episode gradient steps, discount, phase durations (s) |
| `[routing]` | `num_cars` | Vehicles generated per episode into `.rou.xml` |
| `[dir]` | `model_folder`, `sumo_file` | Output folder under `models/`; scenario path under `sumo_files/` without `.sumocfg` |

Example: `sumo_file = 1TL/1TL` → `sumo_files/1TL/1TL.sumocfg`.

### `test_settings.ini`

| Section | Keys | Meaning |
|---------|------|---------|
| `[agent]` | `input_dim` | Must match the trained model’s `input_dim` |
| `[simulation]` | `sumo_gui`, `max_step`, `green_duration`, `yellow_duration` | Inference run |
| `[dir]` | `model_folder`, `sumo_file` | Trained weights location; scenario to simulate |

**Note:** Testing uses the route file already on disk for that scenario; it does not run the same per-episode **Routing** regeneration as training.

## Methodology overview

**Control loop.** SUMO starts via TraCI (`sumo` or `sumo-gui`). For each traffic light, when the current phase duration ends, the code reads **state**, computes **reward** (training only), picks an **action** (green phase index), then applies **yellow** and **green** with `traci.trafficlight.setRedYellowGreenState` and `setPhaseDuration`. Training uses **epsilon-greedy** exploration; `epsilon` decreases linearly over episodes in `train.py`.

**State.** Per TL: a **position matrix** (occupancy grid), **velocity matrix** (normalized speeds), and **phase vector** (indicator for active green phase). Resolution follows `input_dim` and a fixed cell length in `train_simulation.py` / `test_simulation.py`.

**Actions.** Indices over **green phases** (TLS program phases are paired; yellow handled separately). Action count ≈ half the TLS phase count.

**Reward.** Combines waiting-time change with a throughput-style term (vehicles passed vs. queue). See `TrainSimulation.run` in `src/train_simulation.py` for the exact expression.

**Learning.** `Agent` in `src/agent.py` implements a **DQN-style** update: Q prediction, one-step Bellman-style target on the selected action, **MSE** loss, **deque** replay with random minibatches.

## Source code layout

| Path | Purpose |
|------|---------|
| `src/train.py` | Training entry: INI load, episode loop, save models and plots |
| `src/train_simulation.py` | `TrainSimulation`: TraCI loop, state/reward/action, replay |
| `src/test.py` | Test entry: INI load, run test sim, export plots |
| `src/test_simulation.py` | `TestSimulation`: load `.h5`, greedy policy, vehicle stats |
| `src/static_simulation.py` | Static phase rotation; outputs often use `-Static` in filenames |
| `src/agent.py` | `Agent` (train), `TestAgent` (inference) |
| `src/routing.py` | `Routing`: writes training `*.rou.xml` per episode |
| `src/tools.py` | Config, SUMO cmdline, paths, net helpers |
| `src/plot.py` | Matplotlib PNG + `.txt` series |
| `src/logger.py` | `debug.log` |
| `src/create_settings.py` | INI generation helper |
| `src/interface/trafficlight.py` | Per-TL statistics |
| `src/interface/route.py` | Route objects for demand |
| `src/lake.py` | Optional Tkinter GUI (needs assets) |
| `src/testing/routing.py` | Standalone script; not imported by main flow |

## SUMO scenarios

Files live under `sumo_files/`: typically `*.sumocfg`, `*.net.xml`, and `*.rou.xml`. **Training** overwrites `*.rou.xml` via `Routing`. Included layouts include **1TL**, **2TL**, **7TL**, and several **Roxas**-area networks. On **case-sensitive** systems, `sumo_file` must match directory names exactly (e.g. `roxas/All/Roxas` vs `Roxas/All/Roxas`).

## Model artifacts and naming

Saved weights:

```text
model_<input_dim>.<num_lanes>.<output_dim>.h5
```

Example: `model_16.12.4.h5` → `input_dim=16`, 12 lanes (as counted in code), 4 green phases.

Location:

```text
models/<model_folder>/<traffic_light_id>/model_....h5
```

`test_simulation.py` expects this naming; keep `input_dim` and architecture aligned with training.

## Outputs and logging

**Training:** `*.h5` per TL; `plot_reward.png`, `plot_delay.png`, `plot_queue.png` and `plot_*_data.txt`; copy of `train_settings.ini` under `models/<model_folder>/`.

**Testing / static:** `test_*_data.txt` series; bar-chart PNGs; static runs may use `-Static` in names.

**Logs:** `debug.log` from `src/logger.py`.

## Evaluation metrics

The code aggregates **queue** and **waiting**-related quantities (cumulative waiting, average queue length) for plots and exports. For publications, consider also logging **per-vehicle delay**, **throughput**, and **fairness**—these need small TraCI extensions.

## Optional GUI (`lake.py`)

`src/lake.py` is a **Tkinter** launcher that can write `test_settings.ini` and run tests. It expects images under `lake_gui/` (e.g. `lake-bg-info.png`). If that folder is absent, use the CLI commands above.

## Troubleshooting

| Symptom | Likely cause | What to try |
|---------|----------------|-------------|
| `please declare environment variable 'SUMO_HOME'` | Missing env | Set `SUMO_HOME`; restart shell |
| `Model not found!` | Wrong path or name | Match `model_folder` and `model_*.h5` layout |
| `traci` / `sumolib` import errors | Wrong env or SUMO | Same venv as `pip install -r requirements.txt`; valid `SUMO_HOME/tools` |
| Crash or empty run | Bad `sumo_file` | Confirm `sumo_files/<path>.sumocfg` exists; check case on Linux |
| Very slow training | GUI, long `max_step`, CPU | `sumo_gui = False`; shorten steps/episodes while debugging |
| OOM | Large buffer / batch | Lower `size_max` or `batch_size` |

**Speed tips:** Disable GUI for long runs; reduce `total_episodes` and `max_step` until stable; use GPU TensorFlow if available; moderate `num_cars` during first tests.

## Roadmap and future work

- First-class **throughput** and **travel-time** metrics in tests.
- **Curriculum** or **transfer** across scenarios.
- **Multi-agent** coordination.
- **Reward** ablations and sensitivity notes.
- **CI** smoke test: headless SUMO on a tiny network.

## References

- [SUMO](https://www.eclipse.org/sumo/)
- [TraCI](https://sumo.dlr.de/docs/TraCI.html)
- [TensorFlow / Keras](https://www.tensorflow.org/)

## License

See [`LICENSE`](LICENSE) in the repository root.

## Contributing

Issues and pull requests are welcome. For bugs, include OS, Python version, SUMO version, relevant INI snippets, reproduction steps, and `debug.log` excerpts (no secrets). For features, describe the metric or scenario and how it should hook into the TraCI loop.
