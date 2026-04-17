# Designing-and-Analysing-a-Fair-Variant-of-Cinquillo

**Configurable Python Engine with Flask Web Interface**

Cinquillo 2.0 is a configurable implementation of the game designed for both interactive play and experimental research.  
It includes:

- A modular Python game engine  
- A Flask-based web interface  
- Multiple AI agents (Random, Heuristic, MCTS, RL)  
- A tournament/simulation framework  
- Full project documentation  

---

# Cinquillo 2.0

Cinquillo 2.0 is a research platform for installing, configuring, and operating multiple components of a board-game analysis system, including:

- the game engine
- the simulation pipeline
- the reinforcement learning training script
- the tournament harness
- the interactive web interface

---

## Prerequisites

The platform was developed and tested on **macOS 14**.

Make sure the following software is installed:

- **Python 3.10+**
- **pip 25.2+**
- **venv** for virtual environments
- **Git** for cloning the repository
- A modern web browser such as:
  - Chrome
  - Firefox
  - Safari

### Main Python Packages

| Package | Purpose |
|--------|---------|
| `numpy` | Array operations and RL state encoding |
| `torch` | Double-DQN training |
| `flask` | Lightweight web server |
| `flask-cors` | Cross-origin resource sharing for the UI |
| `matplotlib` | Figure generation |
| `scipy` | Statistical tests |
| `statsmodels` | Regression and ANOVA helpers |
| `tqdm` | Progress bars during simulation runs |
| `pytest` | Automated test suite |

---

## Installation

Clone the repository and set up the environment:

```bash
git clone https://github.com/Ale-Newport/Designing-and-Analysing-a-Fair-Variant-of-Cinquillo.git
# Or download the source code manually

cd Designing-and-Analysing-a-Fair-Variant-of-Cinquillo
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
````

---

## Project Layout

```text
PRJ/
├── game/
│   ├── entities.py              # Card, Deck, Board, GameState
│   └── rules.py                 # Move types, Rules
├── agents/
│   ├── base_agents.py           # RandomAgent, HeuristicAgent, factory functions
│   ├── mcts_agent.py            # MCTSAgent + four speed variants
│   └── rl_agent.py              # RLAgent (Double-DQN, 209-feature encoder)
├── rl_agent/
│   ├── train_agent.py           # RL training (three-stage curriculum)
│   └── models/
│       └── rl_agent.pkl         # Saved checkpoint
├── simulation/
│   ├── run_simulations.py       # 16-experiment simulation runner
│   ├── visualise_results.py     # Figure and LaTeX table generator
│   ├── visualise_flow.py        # Per-game turn-flow diagram generator
│   └── tournament.py            # Round-robin tournament harness
├── web_server.py                # Flask back-end
├── templates/
│   └── cinquillo.html           # Single-page front-end
├── output/
│   ├── data/                    # Generated JSON/CSV simulation results
│   └── figures/                 # Generated PNG/PDF figures
└── requirements.txt
```

---

## Running the Test Suite

Run the full automated test suite from the project root:

```bash
pytest tests/ -v
```

Expected result:

```bash
260 passed
```

(Some tests may be skipped depending on the environment.)

---

## Running Simulations

### Full simulation run

```bash
python simulation/run_simulations.py
```

### Quick or medium mode

```bash
python simulation/run_simulations.py --quick
python simulation/run_simulations.py --medium
```

### Skip slow agents during development

```bash
python simulation/run_simulations.py --skip-mcts
python simulation/run_simulations.py --skip-rl
```

### Run specific experiments only

Example for experiments 1, 8, and 9:

```bash
python simulation/run_simulations.py --exp 1 8 9
```

### Output locations

Simulation results are written to:

* `output/data/experiments/` for JSON aggregates
* `output/data/raw/` for per-game CSV rows

---

## Generating Figures and Tables

Generate all figures and LaTeX tables:

```bash
python simulation/visualise_results.py
```

Generate figures for specific experiments only:

```bash
python simulation/visualise_results.py --exp 1 8 9
```

---

## Generating Game-Trace Flow Diagrams

Generate an annotated game-trace diagram:

```bash
python simulation/visualise_flow.py "baseline"
```

Example variant names (case-insensitive):

* `baseline`
* `card-flood`
* `double-edge`
* `lucky-draw`
* `chaos-mode`

The output is a self-contained `.tex` file that can be compiled independently with `pdflatex`.

---

## Training the RL Agent

From the project root:

```bash
cd PRJ
python rl_agent/train_agent.py
```

Best checkpoint is saved to:

```text
rl_agent/models/rl_agent.pkl
```

A new checkpoint is written whenever a better mean win rate is achieved.

### Training curriculum

The RL training process uses three stages:

* **Stage 0**: random opponents
* **Stage 1**: one heuristic agent added
* **Stage 2**: full heuristics across all 16 variants

The learning rate schedule and gradient update cap activate at **episode 10,000**.

---

## Starting the Web Interface

Launch the Flask web server:

```bash
python web_server.py
```

Then open:

```text
http://localhost:4000
```

in any modern browser.

---

## Web Interface Features

The web interface allows a human player to compete against any combination of AI agents in a configurable variant.

### Main controls

* **New Game** — deals a fresh hand and resets the board
* **Agent selector** — choose the AI type for each seat:

  * Random
  * Aggressive
  * Defensive
  * Balanced
  * MCTS
  * RL
* **Variant selector** — switch between named variants or build a custom configuration
* **Card area** — click a highlighted card to play it, or use the **Roll Dice** / **Pass** buttons
* **Board area** — shows the four suit sequences built so far, colour-coded by suit



---

## Summary

Cinquillo 2.0 provides a complete platform for:

* game execution
* simulation experiments
* statistical analysis
* RL training
* tournament evaluation
* interactive play through a browser interface


