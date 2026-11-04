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

from collections.abc import Callable
import contextlib
import math
import threading
from typing import Any

from builtin_interfaces.msg import Time
from geometry_msgs.msg import Twist
import gymnasium as gym
from nav_msgs.msg import Odometry
import numpy as np
import rclpy
from rclpy.client import Client
from rclpy.context import Context
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from ros_gz_interfaces.srv import ControlWorld
from rosgraph_msgs.msg import Clock

from goal_reaching import task


def to_ns(stamp: Time) -> int:
    """Return a ROS time stamp in integer nanoseconds (exact, no float rounding)."""
    return stamp.sec * 1_000_000_000 + stamp.nanosec


def odometry_state(msg: Odometry) -> tuple[float, float, float, float, float]:
    """Extract ``(x, y, yaw, v, w)`` from an odometry message.

    The orientation comes as a quaternion; for a robot on flat ground only the
    rotation about z matters, and this is the standard formula for its angle.
    """
    q = msg.pose.pose.orientation
    yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
    return (
        msg.pose.pose.position.x,
        msg.pose.pose.position.y,
        yaw,
        msg.twist.twist.linear.x,
        msg.twist.twist.angular.z,
    )


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
        self.control: Client = self.create_client(ControlWorld, f'/world/{world}/control')

    def _store(self, key: str, msg: Any) -> None:
        with self._changed:
            self._last[key] = msg
            self._count[key] += 1
            self._changed.notify_all()

    def count(self, key: str) -> int:
        """Return how many messages of key have arrived so far."""
        with self._changed:
            return self._count[key]

    def wait(self, key: str, since: int, predicate: Callable[[Any], bool], timeout: float) -> Any:
        """Block until a message of key newer than number `since` satisfies predicate.

        Args:
            key: 'odom' or 'clock'.
            since: Value of ``count(key)`` taken before the action; only later
                messages are accepted.
            predicate: Question about the message, e.g. "is its stamp late enough?".
            timeout: Seconds of wall time before giving up.

        Returns:
            The message that satisfied the predicate.

        Raises:
            TimeoutError: If no such message arrived in time.
        """
        with self._changed:
            ok = self._changed.wait_for(
                lambda: self._count[key] > since and predicate(self._last[key]), timeout
            )
            if not ok:
                raise TimeoutError(f'no new {key} message in {timeout} s')
            return self._last[key]

    def call_control(self, request: ControlWorld.Request, timeout: float) -> None:
        """Send a world control request and wait for Gazebo's answer.

        The answer only means that Gazebo accepted the request; for a multi_step
        it arrives before the steps are done, so callers wait for messages too.
        """
        future = self.control.call_async(request)
        done = threading.Event()
        future.add_done_callback(lambda _: done.set())  # runs in the ROS thread
        if not done.wait(timeout):
            raise TimeoutError(f'{self.control.srv_name} did not answer in {timeout} s')
        if not future.result().success:
            raise RuntimeError(f'{self.control.srv_name} rejected the request')


class GoalReachingRosEnv(gym.Env):
    """Drive the diff-drive robot to a goal, with the simulation behind ROS 2.

    Observation and reward come from :mod:`goal_reaching.task`. The action is
    ``[linear, angular]`` in [-1, 1], scaled to the velocity limits of the robot.

    Stepped mode: the world stays paused and the environment advances it exactly
    one control period per step. Start the simulation with::

        ros2 launch goal_reaching sim.launch.py paused:=true real_time_factor:=0
    """

    metadata = {'render_modes': []}

    def __init__(
        self,
        namespace: str = '',
        world: str = 'goal_reaching',
        physics_step: float = 0.001,
        timeout: float = 5.0,
    ) -> None:
        """Start the ROS 2 node in a background thread.

        Nothing is sent to the simulation here, so the environment can be created
        (and closed) even when the simulation is not running yet.

        Args:
            namespace: Namespace of the simulation's topics (``namespace`` argument
                of the launch file).
            world: Name of the Gazebo world, part of the control service name.
            physics_step: ``<max_step_size>`` of the world (s); with the control
                period it gives the iterations per step (0.1 / 0.001 = 100).
            timeout: Wall-clock seconds to wait for the simulation before failing.
        """
        super().__init__()
        self.observation_space = gym.spaces.Box(
            task.OBSERVATION_LOW, task.OBSERVATION_HIGH, dtype=np.float32
        )
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self._iterations = round(task.CONTROL_PERIOD / physics_step)
        self._timeout = timeout
        self._goal = np.zeros(2)
        self._previous_distance = 0.0
        self._steps = 0

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

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Put the robot back at the origin, draw a new goal and observe.

        Returns:
            The first observation and an info dict with the goal and the
            simulation time of the observation.
        """
        super().reset(seed=seed)
        if not self._ros.control.wait_for_service(timeout_sec=self._timeout):
            raise RuntimeError(
                f'{self._ros.control.srv_name} is not available. Is the simulation running? '
                'ros2 launch goal_reaching sim.launch.py paused:=true real_time_factor:=0'
            )

        # 'reset all' alone would also unpause the world: send both in one request.
        since = self._ros.count('odom')
        since_clock = self._ros.count('clock')
        request = ControlWorld.Request()
        request.world_control.pause = True
        request.world_control.reset.all = True
        self._ros.call_control(request, self._timeout)
        # Gazebo answers before applying the reset, and a control request sent
        # before it is applied replaces it: the reset would be lost. Wait until a
        # new /clock shows the time back at zero.
        self._ros.wait('clock', since_clock, lambda m: to_ns(m.clock) == 0, self._timeout)

        # A paused world publishes no odometry, so advance one control period (the
        # command was cleared by the reset) and wait for the odometry at its end.
        request = ControlWorld.Request()
        request.world_control.pause = True
        request.world_control.multi_step = self._iterations
        self._ros.call_control(request, self._timeout)
        target = round(task.CONTROL_PERIOD * 1e9)
        odom = self._ros.wait(
            'odom', since, lambda m: to_ns(m.header.stamp) >= target, self._timeout
        )

        self._goal = task.sample_goal(self.np_random)
        x, y, yaw, v, w = odometry_state(odom)
        self._previous_distance = task.distance_to_goal(x, y, self._goal)
        self._steps = 0
        info = {'goal': self._goal.copy(), 'sim_time': to_ns(odom.header.stamp) / 1e9}
        return task.observation(x, y, yaw, v, w, self._goal), info

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
