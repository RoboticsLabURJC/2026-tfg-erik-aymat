"""Tests of the environment that need ROS 2 but no simulation running."""

import math

from goal_reaching.ros_env import GoalReachingRosEnv
import numpy as np
import pytest

from goal_reaching import task


@pytest.fixture
def env():
    environment = GoalReachingRosEnv()
    yield environment
    environment.close()


def test_spaces(env):
    assert env.observation_space.shape == (task.OBSERVATION_SIZE,)
    assert env.observation_space.dtype == np.float32
    assert env.action_space.shape == (2,)
    assert env.action_space.dtype == np.float32
    np.testing.assert_array_equal(env.action_space.low, [-1.0, -1.0])
    np.testing.assert_array_equal(env.action_space.high, [1.0, 1.0])


def test_worst_case_observation_fits_the_space(env):
    # Robot in a corner of the arena, overshot by one step at full speed, moving at
    # its limits; goal at the maximum distance on the opposite side.
    corner = task.ARENA_HALF_SIZE + task.MAX_LINEAR_SPEED * task.CONTROL_PERIOD
    far_goal = -task.GOAL_MAX_DISTANCE * np.array([math.sqrt(0.5), math.sqrt(0.5)])
    obs = task.observation(
        corner, corner, 0.0, task.MAX_LINEAR_SPEED, -task.MAX_ANGULAR_SPEED, far_goal
    )
    assert env.observation_space.contains(obs)


def test_close_twice_is_harmless(env):
    env.close()
    env.close()


def test_two_environments_shut_down_independently():
    first = GoalReachingRosEnv(namespace='sim1')
    second = GoalReachingRosEnv(namespace='sim2')
    first.close()
    assert second._context.ok()
    second.close()
    assert not second._context.ok()
