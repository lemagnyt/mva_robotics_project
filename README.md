# MVA Robotics Project

This project implements a **robust recovery controller** for a quadrupedal robot, inspired by the paper:

**"Robust Recovery Controller for a Quadrupedal Robot using Deep Reinforcement Learning"**
Joonho Lee, Jemin Hwangbo, Marco Hutter (2019)
[IEEE / arXiv](https://arxiv.org/abs/1901.07517)

The goal is to enable a quadrupedal robot (ANYmal) to **stand up from any fallen position and continue walking**, using deep reinforcement learning (Deep RL) and a hierarchical behavior-based controller.

---

## Project Structure

```
mva_robotics_project/
├── mujoco_menagerie/       # Submodule from Google DeepMind to load different robot models (we use anymal_b)
├── utils/
│   ├── robot_loader.py     # Class to load a robot model
│   └── utils_functions.py  # Utility functions used in the project
├── envs/
│   ├── base_env.py         # BaseEnv class (subclass of gym.Env)
│   ├── self_righting_env.py    # Subclass of BaseEnv for the self-righting behavior
│   ├── standing_env.py      # Subclass of BaseEnv for the standing-up behavior
│   ├── locomotion_env.py       # Subclass of BaseEnv for the locomotion behavior
│   └── behavior_selector_env.py  # Class to initialize the Behavior Selector environment
├── train/
│   ├── train_self_righting.py
│   ├── train_standing_up.py
│   ├── train_locomotion.py
│   └── train_behavior_selector.py
├── sim/
│   ├── test_self_righting.py
│   ├── test_standing.py
│   ├── test_locomotion.py
│   ├── test_simple_selector.py  # Test with the naive selector proposed in the paper
│   └── test_deep_selector.py    # Test with the trained Behavior Selector
└── models/                  # Saved trained models
```

### Directory Details

* **`mujoco_menagerie/`**: Contains robot models for Mujoco. We use `anymal_b`.
* **`utils/`**: Utility classes and functions to load the robot and manipulate its data.
* **`envs/`**: Implementation of gym environments for each behavior. Each class defines:

  * `reset()`: reset env and set initial state
  * `get_reward()`: cost function
  * `get_observation()`: agent observations
  * `get_terminated()`: episode termination condition
* **`train/`**: Training scripts for each behavior and the Behavior Selector.
* **`sim/`**: Test scripts to verify each policy and the selector.
* **`models/`**: Folder to save trained models.

---

## Installation and Requirements

This project works with **Python 3.13.9** (other versions untested):

```bash
# Clone the repository with its submodules
git clone --recurse-submodules https://github.com/lemagnyt/mva_robotics_project.git
cd mva_robotics_project

# Install dependencies
pip install -r requirements.txt
```

On **Linux**, some C++ bindings may require preloading `libstdc++`:

```bash
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
```

---

## Useful Commands

### Train a Behavior

```bash
python -m train.train_self_righting
python -m train.train_standing
python -m train.train_locomotion
```

### Train the Behavior Selector

```bash
python -m train.train_behavior_selector
```

### Test a Behavior

```bash
python -m sim.test_self_righting
python -m sim.test_standing
python -m sim.test_locomotion
```

### Test the Behavior Selector

```bash
python -m sim.test_simple_selector   # Naive selector
python -m sim.test_deep_selector    # Trained selector
```

---

## Concept

The project implements a **hierarchical behavior-based controller**:

1. **Self-righting**: brings the robot upright after a fall.
2. **Standing-up**: moves from a sitting to a standing posture.
3. **Locomotion**: follows velocity commands.

Each behavior is trained individually in simulation (TRPO + GAE) and tested on the robot. Then, a **Behavior Selector** learns to choose the appropriate behavior based on the current state and action history, ensuring smooth and robust transitions.
