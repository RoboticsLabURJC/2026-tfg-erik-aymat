"""Start the goal-reaching simulation.

Gazebo runs the world, which already contains the robot (included from its URDF).
On the ROS side there are three nodes: the bridge, robot_state_publisher (TF of
the robot body) and, optionally, RViz. The ROS nodes can be pushed into a
namespace to run several simulations in parallel; Gazebo itself is isolated with
GZ_PARTITION, which is an environment variable and not a launch argument.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchContext, LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    GroupAction,
    IncludeLaunchDescription,
    OpaqueFunction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, PushROSNamespace


def gazebo(context: LaunchContext) -> list[IncludeLaunchDescription]:
    """Build the gz sim command line from the launch arguments."""
    args = []
    if LaunchConfiguration('headless').perform(context) == 'true':
        args += ['-s', '--headless-rendering']  # server only, sensors still render
    if LaunchConfiguration('paused').perform(context) != 'true':
        args.append('-r')  # run on start; without it the world waits paused
    args.append(LaunchConfiguration('world').perform(context))

    gz_sim_launch = os.path.join(
        get_package_share_directory('ros_gz_sim'), 'launch', 'gz_sim.launch.py'
    )
    return [
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(gz_sim_launch),
            # gz_sim.launch.py also adds the paths exported in package.xml to
            # GZ_SIM_RESOURCE_PATH, which is how model://goal_reaching/... is found.
            launch_arguments={'gz_args': ' '.join(args), 'on_exit_shutdown': 'true'}.items(),
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
            'use_sim_time',
            default_value='true',
            description='Use the Gazebo clock (/clock) in the ROS nodes.',
        ),
        DeclareLaunchArgument('rviz', default_value='false', description='Open RViz.'),
        DeclareLaunchArgument(
            'namespace', default_value='', description='Namespace for the ROS nodes and topics.'
        ),
    ]

    # tf2 publishes and subscribes to the absolute /tf and /tf_static. Remapping
    # them to relative names keeps each namespaced simulation with its own TF.
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
