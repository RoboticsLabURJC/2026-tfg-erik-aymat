"""Goal-reaching task logic, independent of ROS and Gazebo.

Pure functions shared by the three environments (stepped, real-time and direct
Gazebo), so all of them solve exactly the same task. They take plain numbers,
which also makes them testable without a simulator.

Units are SI; poses follow REP-103 (x forward, y left, yaw counter-clockwise).
"""

import math

import numpy as np

# Goals are drawn on a ring around the robot's start pose (the world origin).
GOAL_MIN_DISTANCE = 0.5  # m
GOAL_MAX_DISTANCE = 2.0  # m

# Velocity limits of the DiffDrive plugin (urdf/diff_drive_robot.urdf); used to
# scale the measured velocities to [-1, 1].
MAX_LINEAR_SPEED = 0.2  # m/s
MAX_ANGULAR_SPEED = 1.0  # rad/s

OBSERVATION_SIZE = 5


def sample_goal(rng: np.random.Generator) -> np.ndarray:
    """Draw a goal position on the ring around the origin.

    Distance and direction are uniform, so every distance between the minimum
    and the maximum is equally likely.

    Args:
        rng: Generator of the environment (``self.np_random``), so the goals are
            reproducible from the seed.

    Returns:
        Goal position ``[x, y]`` in metres, in the world frame.
    """
    distance = rng.uniform(GOAL_MIN_DISTANCE, GOAL_MAX_DISTANCE)
    direction = rng.uniform(-math.pi, math.pi)
    return np.array([distance * math.cos(direction), distance * math.sin(direction)])


def observation(
    x: float, y: float, yaw: float, v: float, w: float, goal: np.ndarray
) -> np.ndarray:
    """Build the agent observation from the robot state and the goal.

    The goal is given relative to the robot (distance and heading error), so the
    policy does not depend on where the episode happens in the world. The heading
    error enters as sine and cosine: an angle jumps from +pi to -pi when the goal
    passes behind the robot, while sine and cosine change smoothly.

    Args:
        x: Robot position along x (m).
        y: Robot position along y (m).
        yaw: Robot heading (rad).
        v: Measured linear velocity (m/s).
        w: Measured angular velocity (rad/s).
        goal: Goal position ``[x, y]`` (m).

    Returns:
        ``[distance / GOAL_MAX_DISTANCE, sin(error), cos(error), v / MAX_LINEAR_SPEED,
        w / MAX_ANGULAR_SPEED]`` as float32. The scaled distance is 1 for the
        farthest goal and can go above 1 if the robot moves away.
    """
    dx = goal[0] - x
    dy = goal[1] - y
    # No need to wrap the angle to [-pi, pi]: only its sine and cosine are used.
    heading_error = math.atan2(dy, dx) - yaw
    return np.array(
        [
            math.hypot(dx, dy) / GOAL_MAX_DISTANCE,
            math.sin(heading_error),
            math.cos(heading_error),
            v / MAX_LINEAR_SPEED,
            w / MAX_ANGULAR_SPEED,
        ],
        dtype=np.float32,
    )
