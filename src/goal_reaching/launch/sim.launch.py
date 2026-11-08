"""Start the goal-reaching simulation.

Gazebo runs the world, which already contains the robot (included from its URDF).
On the ROS side there are three nodes: the bridge, robot_state_publisher (TF of
the robot body) and, optionally, RViz.

The ROS nodes and topics can be pushed into a namespace, but a namespace does not
isolate simulations: /clock and the world control service stay at the root, and
a second Gazebo with the same world in the same partition does not simulate. To
run simulations in parallel, give each one its own GZ_PARTITION and
ROS_DOMAIN_ID (environment variables, not launch arguments).
"""

import os
import tempfile
from xml.etree import ElementTree

from ament_index_python.packages import get_package_share_directory
from launch import LaunchContext, LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    GroupAction,
    OpaqueFunction,
    Shutdown,
)
from launch.conditions import IfCondition, evaluate_condition_expression
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, PushROSNamespace


def with_real_time_factor(world: str, real_time_factor: str) -> str:
    """Write a copy of the world with another real-time factor and return its path.

    The real-time factor lives in the world file and gz sim cannot override it
    exactly: its -z option sets an update rate, and 0 there is ignored instead of
    meaning "no limit". The copy goes to the temporary directory under a name that
    depends on the factor, so launching twice with the same value reuses it.
    """
    tree = ElementTree.parse(world)
    elements = list(tree.getroot().iter('real_time_factor'))
    if not elements:
        raise RuntimeError(f'{world} has no <real_time_factor> to override')
    for element in elements:
        element.text = real_time_factor
    name = os.path.splitext(os.path.basename(world))[0]
    path = os.path.join(tempfile.gettempdir(), f'{name}_rtf_{real_time_factor}.sdf')
    tree.write(path)
    return path


def gazebo(context: LaunchContext) -> list[ExecuteProcess]:
    """Build the gz sim command line from the launch arguments."""
    world = LaunchConfiguration('world').perform(context)
    real_time_factor = LaunchConfiguration('real_time_factor').perform(context)
    if real_time_factor:
        try:
            valid = float(real_time_factor) >= 0
        except ValueError:
            valid = False
        if not valid:
            raise ValueError(f'real_time_factor must be a number >= 0, got {real_time_factor!r}')
        world = with_real_time_factor(world, real_time_factor)

    # Same rules as IfCondition (used for rviz): true/false or 1/0, any case;
    # anything else is an error instead of silently meaning false.
    def is_true(name: str) -> bool:
        return evaluate_condition_expression(context, [LaunchConfiguration(name)])

    args = []
    if is_true('headless'):
        args += ['-s', '--headless-rendering']  # server only, sensors still render
    if not is_true('paused'):
        args.append('-r')  # run on start; without it the world waits paused
    args.append(world)

    # model://goal_reaching/... is resolved through GZ_SIM_RESOURCE_PATH, which must
    # contain the directory holding the package's share folder.
    share_parent = os.path.dirname(get_package_share_directory('goal_reaching'))
    resource_path = os.pathsep.join(
        filter(None, [share_parent, os.environ.get('GZ_SIM_RESOURCE_PATH', '')])
    )
    # gz is started directly, not through ros_gz_sim's gz_sim.launch.py: that one
    # runs it in a shell, and when launch shuts down (launch_testing, a script)
    # only the shell gets the signal, leaving the Gazebo server orphaned.
    return [
        ExecuteProcess(
            cmd=['gz', 'sim', *args],
            name='gazebo',
            output='screen',
            additional_env={'GZ_SIM_RESOURCE_PATH': resource_path},
            on_exit=Shutdown(),  # closing the GUI ends the whole launch
        )
    ]


def generate_launch_description() -> LaunchDescription:
    """Declare the arguments and the processes of the simulation."""
    share = get_package_share_directory('goal_reaching')
    # Plain URDF (no xacro), so reading the file is enough. It must be the same
    # file the world includes, or RViz and Gazebo would show different robots.
    with open(os.path.join(share, 'urdf', 'diff_drive_robot.urdf')) as f:
        robot_description = f.read()
    use_sim_time = LaunchConfiguration('use_sim_time')

    arguments = [
        DeclareLaunchArgument(
            'world',
            default_value=os.path.join(share, 'worlds', 'goal_reaching.sdf'),
            description='World file; it must include the robot.',
        ),
        DeclareLaunchArgument(
            'headless',
            default_value='true',
            description='Run only the Gazebo server, without GUI.',
        ),
        DeclareLaunchArgument(
            'paused',
            default_value='false',
            description='Start with the world paused (stepped mode).',
        ),
        DeclareLaunchArgument(
            'real_time_factor',
            default_value='',
            description='Override the real-time factor of the world: 0 runs as fast as '
            'possible (stepped mode), 1 is real time. Empty keeps the value of the world.',
        ),
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use the Gazebo clock (/clock) in the ROS nodes.',
        ),
        DeclareLaunchArgument('rviz', default_value='false', description='Open RViz.'),
        DeclareLaunchArgument(
            'namespace',
            default_value='',
            description='Namespace for the ROS nodes and topics. It does not isolate '
            'simulations: use GZ_PARTITION and ROS_DOMAIN_ID for that.',
        ),
    ]

    # tf2 publishes and subscribes to the absolute /tf and /tf_static. Remapping
    # them to relative names keeps TF inside the namespace too.
    tf_remappings = [('/tf', 'tf'), ('/tf_static', 'tf_static')]

    ros_nodes = GroupAction(
        [
            PushROSNamespace(LaunchConfiguration('namespace')),
            Node(
                package='ros_gz_bridge',
                executable='parameter_bridge',
                parameters=[{'config_file': os.path.join(share, 'config', 'bridge.yaml')}],
                output='screen',
            ),
            Node(
                package='robot_state_publisher',
                executable='robot_state_publisher',
                parameters=[
                    {'robot_description': robot_description, 'use_sim_time': use_sim_time}
                ],
                remappings=tf_remappings,
                output='screen',
            ),
            Node(
                package='rviz2',
                executable='rviz2',
                arguments=['-d', os.path.join(share, 'config', 'goal_reaching.rviz')],
                parameters=[{'use_sim_time': use_sim_time}],
                remappings=tf_remappings,
                condition=IfCondition(LaunchConfiguration('rviz')),
                output='screen',
            ),
        ]
    )

    return LaunchDescription([*arguments, OpaqueFunction(function=gazebo), ros_nodes])
