"""Gymnasium environment for the goal-reaching task over ROS 2.

The simulation runs in its own processes (``ros2 launch goal_reaching
sim.launch.py``) and this environment talks to it only through ROS 2: odometry
and /clock in, velocity commands and world control requests out.

Threading model. ``reset()`` and ``step()`` are called from the thread of the RL
algorithm, while ROS 2 delivers messages through callbacks. Each environment owns
a ROS 2 context, a node and an executor that spins in a background thread. The
callbacks only store the last message of each topic and wake whoever is
waiting; the algorithm thread never spins and never runs inside a callback, so
it can block waiting for data without deadlocks.
"""

import contextlib
import threading
from typing import Any

from geometry_msgs.msg import Twist
import gymnasium as gym
from nav_msgs.msg import Odometry
import numpy as np
import rclpy
from rclpy.context import Context
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from ros_gz_interfaces.srv import ControlWorld
from rosgraph_msgs.msg import Clock

from goal_reaching import task


class _RosInterface(Node):
    """ROS 2 side of the environment: subscriptions, publisher and client.

    Every topic keeps its last message and a counter of messages received. The
    counter lets the environment ask for a message *newer* than a given point:
    after a reset the last stored message still belongs to the previous episode.
    """

    def __init__(self, context: Context, namespace: str, world: str) -> None:
        """Create the interfaces; topic names are relative to the namespace."""
        super().__init__('goal_reaching_env', namespace=namespace, context=context)
        self._changed = threading.Condition()
        self._last: dict[str, Any] = {'odom': None, 'clock': None}
        self._count = {'odom': 0, 'clock': 0}
        self.create_subscription(Odometry, 'odom', lambda m: self._store('odom', m), 10)
        self.create_subscription(Clock, '/clock', lambda m: self._store('clock', m), 10)
        self.cmd_vel = self.create_publisher(Twist, 'cmd_vel', 10)
        self.control = self.create_client(ControlWorld, f'/world/{world}/control')

    def _store(self, key: str, msg: Any) -> None:
        with self._changed:
            self._last[key] = msg
            self._count[key] += 1
            self._changed.notify_all()

    def count(self, key: str) -> int:
        """Return how many messages of key have arrived so far."""
        with self._changed:
            return self._count[key]


class GoalReachingRosEnv(gym.Env):
    """Drive the diff-drive robot to a goal, with the simulation behind ROS 2.

    Observation and reward come from :mod:`goal_reaching.task`. The action is
    ``[linear, angular]`` in [-1, 1], scaled to the velocity limits of the robot.
    """

    metadata = {'render_modes': []}

    def __init__(self, namespace: str = '', world: str = 'goal_reaching') -> None:
        """Start the ROS 2 node in a background thread.

        Nothing is sent to the simulation here, so the environment can be created
        (and closed) even when the simulation is not running yet.

        Args:
            namespace: Namespace of the simulation's topics (``namespace`` argument
                of the launch file).
            world: Name of the Gazebo world, part of the control service name.
        """
        super().__init__()
        self.observation_space = gym.spaces.Box(
            task.OBSERVATION_LOW, task.OBSERVATION_HIGH, dtype=np.float32
        )
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

        # Own context: several environments can live in one process (SB3 vector
        # environments) and each one shuts down without touching the others. No
        # signal handlers: Ctrl-C must reach the training script as a normal
        # KeyboardInterrupt, not silently shut ROS down under the algorithm.
        self._context = Context()
        rclpy.init(context=self._context, signal_handler_options=SignalHandlerOptions.NO)
        self._ros = _RosInterface(self._context, namespace, world)
        self._executor = SingleThreadedExecutor(context=self._context)
        self._executor.add_node(self._ros)
        self._spin_thread = threading.Thread(target=self._spin, daemon=True)
        self._spin_thread.start()

    def _spin(self) -> None:
        """Body of the ROS 2 thread: run callbacks until close()."""
        # close() shuts the context down and spin() raises: that is how it ends.
        with contextlib.suppress(ExternalShutdownException):
            self._executor.spin()

    def close(self) -> None:
        """Stop the robot and shut ROS 2 down; safe to call more than once."""
        if not self._context.ok():
            return
        self._ros.cmd_vel.publish(Twist())
        # Order matters: shutting the context down makes spin() return; only when
        # the thread has finished is it safe to destroy what it was using.
        rclpy.shutdown(context=self._context)
        self._spin_thread.join(timeout=2.0)
        self._ros.destroy_node()
