# ROS 2 – Gymnasium Bridge

Bachelor's thesis (TFG) of Erik Aymat Urteaga, Degree in Robotics Software Engineering,
Universidad Rey Juan Carlos (URJC). Advisor: José María Cañas.

A ROS 2 package that acts as a generic, configurable bridge between ROS 2 and the
[Gymnasium](https://gymnasium.farama.org/) API, to train and deploy reinforcement learning
policies on simulated and real robots. It is a conceptual successor of `openai_ros` (ROS 1).

## Goals

- **Synchronization** between Gymnasium's `step()` loop and the asynchronous flow of ROS 2:
  a *step* mode (controlled simulation, faster training) and a *real-time* mode with a
  decoupled buffer (inference, real robots).
- **Declarative YAML configuration**: observation and action spaces, topic mapping, execution
  mode and plugins (reward, termination, reset...).
- **Validation** on robots with different morphologies, trained with different algorithms.

## Status

Early development: the bridge itself is not implemented yet. The current work sets up the
environment and small hand-written experiments used to gather requirements.
Progress is logged weekly, in Spanish, at
<https://roboticslaburjc.github.io/2026-tfg-erik-aymat/>.

### Current experiment: `goal_reaching`

A diff-drive robot (URDF) in a Gazebo world and a hand-written Gymnasium environment,
`GoalReachingRosEnv`, that drives it to random goals over ROS 2. Each `step()` advances the
paused simulation exactly 0.1 s of simulated time. It is used to learn how Gymnasium and
ROS 2 should be synchronized before designing the bridge.

```bash
# Simulation for the stepped environment: paused and as fast as possible
ros2 launch goal_reaching sim.launch.py paused:=true real_time_factor:=0

# Unit tests and an integration test that starts the simulation
python -m colcon test --packages-select goal_reaching && colcon test-result --verbose
```

## Requirements

| Component | Version |
|---|---|
| OS | Ubuntu 26.04 |
| ROS 2 | Lyrical Luth |
| Simulator | Gazebo Jetty, through `ros_gz` |
| Python | 3.14 (the system Python used by ROS 2) |
| GPU | Optional; PyTorch wheels ship CUDA 13 |

## Setup

The Python virtual environment uses `--system-site-packages` so that `rclpy` stays visible,
and `numpy` is pinned to the version shipped by Ubuntu (`constraints.txt`). Creating it needs
the `python3.14-venv` package.

```bash
source /opt/ros/lyrical/setup.bash
python3 -m venv --system-site-packages .venv
touch .venv/COLCON_IGNORE
source .venv/bin/activate
pip install -c constraints.txt -r requirements-dev.txt
pip install -c constraints.txt -r requirements.txt
pre-commit install
python -m colcon build --symlink-install
```

## Repository layout

```
docs/          GitHub Pages site (Spanish): weekly log, decision records, experiments
src/           ROS 2 packages (colcon workspace)
experiments/   training configurations (YAML)
runs/          training outputs (not versioned)
```

## License

Licensed under the [Apache License 2.0](LICENSE).
