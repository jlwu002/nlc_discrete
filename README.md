# Neural Lyapunov Control for Discrete-Time Systems
This repository contains the code for our paper: Junlin Wu, Andrew Clark, Yiannis Kantaros and Yevgeniy Vorobeychik, "Neural Lyapunov Control for Discrete-Time Systems", NeurIPS 2023. 

We designed a novel approach for learning provably verified stabilizing controllers for a broad class of discrete-time nonlinear systems.

<p align="center">
<img src="https://github.com/jlwu002/nlc_discrete/blob/main/poster/nlc_discrete_algo.png" width="700">
</p>

In summary, we make the following contributions:
1. A novel MILP-based approach for verifying a broad class of discrete-time controlled nonlinear systems.
2. A novel approach for learning provably verified stabilizing controllers for a broad class of discrete-time nonlinear systems which combines our MILP-based verifier with a heuristic gradient-based approximate counterexample generation technique.
3. A novel formalization of approximate stability in which the controller provably converges to a small ball near the origin in finite time.
4. Extensive experiments on four standard benchmarks demonstrate that by leveraging the special structure of Lyapunov stability conditions for discrete-time system, our approach significantly outperforms prior art.

# Requirements
To install requirements:

```sh
pip install -r requirements.txt
```

Python 3.7+ and CPLEX version 22.1.0 are required. Run the following commands to check if Python interpreter can successfully locate CPLEX:

```sh
python -c "import cplex"
python -c "from docplex.mp.model import Model"
```

# Train
```sh
python inverted_pendulum.py --seed 0 #inverted pendulum dynamics
python path_tracking.py --seed 0 #path tracking dynamics
python cartpole.py --seed 0 #cartpole dynamics
python pvtol.py --seed 0 #PVTOL dynamics
bash example_run_pendulum.sh #inverted pendulum dynamics seed 0 ~ 9
```

We pretrain the path tracking controller with [stable-baselines3](https://github.com/DLR-RM/stable-baselines3). The gym environment is located [here](https://github.com/jlwu002/nlc_discrete/blob/main/ppo/path_tracking_gymenv.py) and the hyperparameters are located [here](https://github.com/jlwu002/nlc_discrete/blob/main/ppo/PathTracking-v1_8/PathTracking-v1/config.yml).
