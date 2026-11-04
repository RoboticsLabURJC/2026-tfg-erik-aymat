"""Integration test: the stepped environment against a real simulation.

launch_testing starts the simulation with the package's launch file (paused, no
real-time limit), runs the test case and shuts everything down afterwards. It
takes a few seconds, the time Gazebo needs to start.
"""

import os
import unittest

from ament_index_python.packages import get_package_share_directory
from goal_reaching.ros_env import GoalReachingRosEnv
from gymnasium.utils.env_checker import check_env
import launch
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
import launch_testing.actions
import numpy as np
import pytest

from goal_reaching import task


@pytest.mark.launch_test
def generate_test_description():
    sim_launch = os.path.join(
        get_package_share_directory('goal_reaching'), 'launch', 'sim.launch.py'
    )
    simulation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(sim_launch),
        launch_arguments={'paused': 'true', 'real_time_factor': '0'}.items(),
    )
    return launch.LaunchDescription([simulation, launch_testing.actions.ReadyToTest()])


class TestSteppedEnv(unittest.TestCase):
    def setUp(self):
        # Generous timeout: the first reset waits for Gazebo to finish starting.
        self.env = GoalReachingRosEnv(timeout=30.0)

    def tearDown(self):
        self.env.close()

    def test_each_step_advances_one_control_period(self):
        period_ns = round(task.CONTROL_PERIOD * 1e9)
        _, info = self.env.reset(seed=0)
        self.assertEqual(round(info['sim_time'] * 1e9), period_ns)
        for k in range(2, 22):
            obs, _, terminated, truncated, info = self.env.step(np.array([0.5, 0.2]))
            self.assertEqual(round(info['sim_time'] * 1e9), k * period_ns)
            self.assertTrue(self.env.observation_space.contains(obs))
            self.assertFalse(terminated or truncated)

    def test_reset_puts_the_robot_back_at_rest(self):
        self.env.reset(seed=1)
        for _ in range(10):
            self.env.step(np.array([1.0, 1.0]))
        obs, info = self.env.reset(seed=1)
        self.assertAlmostEqual(obs[0], np.linalg.norm(info['goal']) / task.GOAL_MAX_DISTANCE)
        self.assertAlmostEqual(obs[3], 0.0, places=6)
        self.assertAlmostEqual(obs[4], 0.0, places=6)

    def test_gymnasium_env_checker(self):
        check_env(self.env, skip_render_check=True)
