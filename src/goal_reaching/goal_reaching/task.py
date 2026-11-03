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

# Time between two agent actions, in simulated time. Shared by every environment
# so that one step means the same everywhere (100 physics iterations of 1 ms).
CONTROL_PERIOD = 0.1  # s

# Episode end conditions.
SUCCESS_DISTANCE = 0.15  # m; closer than this to the goal counts as reached
ARENA_HALF_SIZE = 3.0  # m; leaving the square [-3, 3] x [-3, 3] ends the episode
MAX_EPISODE_STEPS = 200  # 20 s; the farthest goal needs about 13 s (see the tests)

# Reward terms. Progress dominates: reaching the farthest goal is worth 20 from
# progress plus 10 from the bonus, while 200 steps of penalty only add up to -2.
PROGRESS_REWARD_SCALE = 10.0  # per metre closer to the goal
STEP_PENALTY = 0.01  # per step, so that wandering around is not free
SUCCESS_REWARD = 10.0

# Bounds of the observation, for the environment's observation space. The farthest
# the robot can be from the goal: a corner of the arena, overshot by one step at
# full speed, with the goal at its maximum distance on the opposite side.
_MAX_GOAL_DISTANCE_SEEN = (
    math.sqrt(2) * (ARENA_HALF_SIZE + MAX_LINEAR_SPEED * CONTROL_PERIOD) + GOAL_MAX_DISTANCE
)
OBSERVATION_LOW = np.array([0.0, -1.0, -1.0, -1.0, -1.0], dtype=np.float32)
OBSERVATION_HIGH = np.array(
    [_MAX_GOAL_DISTANCE_SEEN / GOAL_MAX_DISTANCE, 1.0, 1.0, 1.0, 1.0], dtype=np.float32
)


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


def distance_to_goal(x: float, y: float, goal: np.ndarray) -> float:
    """Return the distance from the robot position to the goal (m)."""
    return math.hypot(goal[0] - x, goal[1] - y)


def is_success(distance: float) -> bool:
    """Return whether the robot is close enough to the goal to count as reached."""
    return distance < SUCCESS_DISTANCE


def is_out_of_bounds(x: float, y: float) -> bool:
    """Return whether the robot has left the arena."""
    return abs(x) > ARENA_HALF_SIZE or abs(y) > ARENA_HALF_SIZE


def reward(previous_distance: float, distance: float) -> float:
    """Return the reward of one step.

    The main term rewards progress: getting closer to the goal is positive and
    moving away is negative, so going back and forth earns nothing. A small
    penalty per step makes the shortest path the best one, and a bonus marks the
    success.

    Leaving the arena has no penalty of its own: the progress term has already
    charged for moving away from the goal on the way out.

    Args:
        previous_distance: Distance to the goal before the step (m).
        distance: Distance to the goal after the step (m).

    Returns:
        The reward for this step.
    """
    value = PROGRESS_REWARD_SCALE * (previous_distance - distance) - STEP_PENALTY
    if is_success(distance):
        value += SUCCESS_REWARD
    return value
