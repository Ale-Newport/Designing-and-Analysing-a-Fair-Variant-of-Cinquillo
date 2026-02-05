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

## Core Use Cases

### 1) Interactive play (Web UI)
- Configure dice effects (e.g., `WILD`, `DOUBLE_PLAY`)
- Select scoring mode (`Winner Takes All` or `Double Penalty`)
- Define match end condition (`Target Score` or `Fixed Rounds`)
- Play against AI-controlled opponents

### 2) Simulation and experimentation (Python)
- Evaluate rule variants
- Benchmark agent performance
- Gather large-scale statistics
- Run tournament-style evaluations

### 3) Dissertation/research workflows
- Analyze fairness and balance
- Compare strategy behaviour across agents
- Validate consistency with your HTML implementation
- Produce reproducible experimental results

---


## Project Structure

```text
PRJ/
├── game/              # Core game engine
├── agents/            # AI agents
├── simulation/        # Tournament/simulation framework
├── templates/         # Flask HTML templates
├── web_server.py      # Entry point for web application
├── test_system.py     # Installation/system validation
└── *.md               # Documentation
``` 
