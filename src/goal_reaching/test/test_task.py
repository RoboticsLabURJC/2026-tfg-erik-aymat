import math

from goal_reaching.task import (
    GOAL_MAX_DISTANCE,
    GOAL_MIN_DISTANCE,
    MAX_ANGULAR_SPEED,
    MAX_LINEAR_SPEED,
    OBSERVATION_SIZE,
    observation,
    sample_goal,
)
import numpy as np
import pytest

# --- sample_goal ---------------------------------------------------------------------------


def test_goals_lie_on_the_ring():
    rng = np.random.default_rng(0)
    distances = [np.linalg.norm(sample_goal(rng)) for _ in range(1000)]
    assert min(distances) >= GOAL_MIN_DISTANCE
    assert max(distances) <= GOAL_MAX_DISTANCE


def test_goals_cover_every_direction():
    rng = np.random.default_rng(0)
    goals = np.array([sample_goal(rng) for _ in range(1000)])
    quadrants = {(bool(gx > 0), bool(gy > 0)) for gx, gy in goals}
    assert len(quadrants) == 4


def test_same_seed_gives_same_goals():
    first = [sample_goal(np.random.default_rng(42)) for _ in range(3)]
    second = [sample_goal(np.random.default_rng(42)) for _ in range(3)]
    np.testing.assert_array_equal(first, second)


# --- observation ---------------------------------------------------------------------------


def test_observation_shape_and_type():
    obs = observation(0.0, 0.0, 0.0, 0.0, 0.0, np.array([1.0, 0.0]))
    assert obs.shape == (OBSERVATION_SIZE,)
    assert obs.dtype == np.float32


@pytest.mark.parametrize(
    'goal, sin_error, cos_error',
    [
        ((1.0, 0.0), 0.0, 1.0),  # straight ahead
        ((0.0, 1.0), 1.0, 0.0),  # on the left: turn counter-clockwise
        ((0.0, -1.0), -1.0, 0.0),  # on the right
        ((-1.0, 0.0), 0.0, -1.0),  # behind
    ],
)
def test_heading_error_from_the_origin(goal, sin_error, cos_error):
    obs = observation(0.0, 0.0, 0.0, 0.0, 0.0, np.array(goal))
    assert obs[0] == pytest.approx(1.0 / GOAL_MAX_DISTANCE)
    assert obs[1] == pytest.approx(sin_error, abs=1e-6)
    assert obs[2] == pytest.approx(cos_error, abs=1e-6)


def test_observation_is_relative_to_the_robot():
    # Robot at (1, 1) looking at +y, goal 1 m ahead of it: the same observation as a
    # robot at the origin looking at +x with the goal at (1, 0).
    moved = observation(1.0, 1.0, math.pi / 2, 0.0, 0.0, np.array([1.0, 2.0]))
    at_origin = observation(0.0, 0.0, 0.0, 0.0, 0.0, np.array([1.0, 0.0]))
    np.testing.assert_allclose(moved, at_origin, atol=1e-6)


def test_heading_error_needs_no_wrapping():
    # yaw = 3*pi/2 is the same heading as -pi/2: the observation must not change.
    goal = np.array([1.0, 0.0])
    unwrapped = observation(0.0, 0.0, 3 * math.pi / 2, 0.0, 0.0, goal)
    wrapped = observation(0.0, 0.0, -math.pi / 2, 0.0, 0.0, goal)
    np.testing.assert_allclose(unwrapped, wrapped, atol=1e-6)


def test_velocities_are_scaled_by_their_limits():
    obs = observation(0.0, 0.0, 0.0, MAX_LINEAR_SPEED, -MAX_ANGULAR_SPEED, np.array([1.0, 0.0]))
    assert obs[3] == pytest.approx(1.0)
    assert obs[4] == pytest.approx(-1.0)
