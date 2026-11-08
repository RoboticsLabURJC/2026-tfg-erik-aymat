"""Conversion of ROS messages to the plain numbers task.py works with."""

import math

from builtin_interfaces.msg import Time
from goal_reaching.ros_env import odometry_state, to_ns
from nav_msgs.msg import Odometry
import pytest


def odometry(x, y, yaw, v, w):
    msg = Odometry()
    msg.pose.pose.position.x = x
    msg.pose.pose.position.y = y
    # Rotation of yaw about z as a quaternion.
    msg.pose.pose.orientation.z = math.sin(yaw / 2)
    msg.pose.pose.orientation.w = math.cos(yaw / 2)
    msg.twist.twist.linear.x = v
    msg.twist.twist.angular.z = w
    return msg


@pytest.mark.parametrize('yaw', [0.0, 0.5, math.pi / 2, -math.pi / 2, 3.0, -3.0])
def test_yaw_comes_back_from_the_quaternion(yaw):
    assert odometry_state(odometry(0.0, 0.0, yaw, 0.0, 0.0))[2] == pytest.approx(yaw)


def test_position_and_velocities_are_copied():
    assert odometry_state(odometry(1.5, -2.0, 0.0, 0.1, -0.4)) == pytest.approx(
        (1.5, -2.0, 0.0, 0.1, -0.4)
    )


def test_stamp_in_nanoseconds_is_exact():
    assert to_ns(Time(sec=2, nanosec=400_000_000)) == 2_400_000_000
