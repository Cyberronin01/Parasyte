# PARASYTE - SITL Codebase (Reviewer Replication Package)

Simulation testbed and cognitive architecture for an 8-legged Theo Jansen linkage walker.

## Quickstart

```bash
pip install -r requirements.txt
```

## Running Experiments

### 1. Master Benchmarks (Table 1 & Figs. 5, 6, 7)
Executes Scenarios 1, 2, and 3, prints the master scorecard, and plots 2D trajectories:
```bash
python run_benchmarks.py
```

### 2. Continual Learning Stress Test (Fig. 3)
Compares straight-line gait retention under Elastic Weight Consolidation (EWC) vs. unregularized baseline:
```bash
python run_ewc_experiment.py
```

### 3. Collision Anomaly & Motor Cutoff (Fig. 4)
Simulates mechanical arrest, prediction surprise spike, and 20 ms emergency shutdown:
```bash
python run_anomaly_test.py
```

### 4. Linkage Kinematics & Duty Cycle (Fig. 1)
Solves the closed-loop 11-bar loop closure and generates the foot trajectory curve:
```bash
python run_kinematics_test.py
```

All generated plots and logs are saved directly into `results/`.
