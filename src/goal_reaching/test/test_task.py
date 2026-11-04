import math

from goal_reaching.task import (
    ARENA_HALF_SIZE,
    CONTROL_PERIOD,
    GOAL_MAX_DISTANCE,
    GOAL_MIN_DISTANCE,
    MAX_ANGULAR_SPEED,
    MAX_EPISODE_STEPS,
    MAX_LINEAR_SPEED,
    OBSERVATION_SIZE,
    PROGRESS_REWARD_SCALE,
    STEP_PENALTY,
    SUCCESS_DISTANCE,
    SUCCESS_REWARD,
    action_to_velocity,
    distance_to_goal,
    is_out_of_bounds,
    is_success,
    observation,
    reward,
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


# --- end conditions ------------------------------------------------------------------------


def test_distance_to_goal():
    assert distance_to_goal(1.0, 1.0, np.array([4.0, 5.0])) == pytest.approx(5.0)


def test_success_threshold_is_strict():
    assert is_success(SUCCESS_DISTANCE - 1e-3)
    assert not is_success(SUCCESS_DISTANCE)


@pytest.mark.parametrize(
    'x, y, out',
    [
        (0.0, 0.0, False),
        (ARENA_HALF_SIZE, -ARENA_HALF_SIZE, False),  # on the border is still inside
        (ARENA_HALF_SIZE + 0.01, 0.0, True),
        (0.0, -ARENA_HALF_SIZE - 0.01, True),
    ],
)
def test_out_of_bounds(x, y, out):
    assert is_out_of_bounds(x, y) == out


def test_every_goal_is_inside_the_arena():
    assert GOAL_MAX_DISTANCE < ARENA_HALF_SIZE


def test_episode_is_long_enough_for_the_farthest_goal():
    # Worst case: the farthest goal right behind the robot. Turn half a circle, then
    # drive straight at full speed (acceleration limits ignored: they add < 0.5 s).
    seconds = math.pi / MAX_ANGULAR_SPEED + GOAL_MAX_DISTANCE / MAX_LINEAR_SPEED
    assert seconds / CONTROL_PERIOD < MAX_EPISODE_STEPS


# --- reward --------------------------------------------------------------------------------


def test_reward_for_getting_closer():
    assert reward(1.0, 0.9) == pytest.approx(0.1 * PROGRESS_REWARD_SCALE - STEP_PENALTY)


def test_reward_for_moving_away():
    assert reward(0.9, 1.0) == pytest.approx(-0.1 * PROGRESS_REWARD_SCALE - STEP_PENALTY)


def test_standing_still_only_pays_the_step_penalty():
    assert reward(1.0, 1.0) == pytest.approx(-STEP_PENALTY)


def test_going_back_and_forth_earns_nothing():
    assert reward(1.0, 0.8) + reward(0.8, 1.0) == pytest.approx(-2 * STEP_PENALTY)


def test_reaching_the_goal_adds_the_bonus():
    expected = 0.1 * PROGRESS_REWARD_SCALE - STEP_PENALTY + SUCCESS_REWARD
    assert reward(0.2, 0.1) == pytest.approx(expected)


# --- action --------------------------------------------------------------------------------


def test_action_is_scaled_to_the_velocity_limits():
    assert action_to_velocity(np.array([1.0, -1.0])) == pytest.approx(
        (MAX_LINEAR_SPEED, -MAX_ANGULAR_SPEED)
    )
    assert action_to_velocity(np.array([0.5, 0.0])) == pytest.approx((MAX_LINEAR_SPEED / 2, 0.0))


def test_action_outside_the_range_is_clipped():
    assert action_to_velocity(np.array([3.0, -2.0])) == pytest.approx(
        (MAX_LINEAR_SPEED, -MAX_ANGULAR_SPEED)
    )


@pytest.mark.parametrize('bad', [math.nan, math.inf])
def test_non_finite_action_is_an_error(bad):
    with pytest.raises(ValueError, match='finite'):
        action_to_velocity(np.array([bad, 0.0]))
