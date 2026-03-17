# Copyright (c) 2024，Horizon Robotics.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import IncludeLaunchDescription
from launch_ros.actions import Node
from launch.substitutions import TextSubstitution
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python import get_package_share_directory
from ament_index_python.packages import get_package_prefix


def generate_launch_description():
    # 拷贝config中文件
    dnn_node_example_path = os.path.join(
        get_package_prefix('dnn_node_example'),
        "lib/dnn_node_example")
    print("dnn_node_example_path is ", dnn_node_example_path)
    cp_cmd = "cp -r " + dnn_node_example_path + "/config ."
    print("cp_cmd is ", cp_cmd)
    os.system(cp_cmd)

    # 启动参数（仅保留双路链路需要的最小集合）
    config_file_launch_arg = DeclareLaunchArgument(
        "dnn_example_config_file",
        default_value=TextSubstitution(text="config/yolo11workconfig.json")
    )
    dump_render_launch_arg = DeclareLaunchArgument(
        "dnn_example_dump_render_img",
        default_value=TextSubstitution(text="0")
    )
    msg_pub_topic_name_launch_arg = DeclareLaunchArgument(
        "dnn_example_msg_pub_topic_name",
        default_value=TextSubstitution(text="hobot_dnn_detection")
    )
    cam0_device_launch_arg = DeclareLaunchArgument(
        "cam0_device",
        default_value=TextSubstitution(
            text="/dev/v4l/by-path/platform-xhci-hcd.2.auto-usb-0:1.1:1.0-video-index0")
    )
    cam1_device_launch_arg = DeclareLaunchArgument(
        "cam1_device",
        default_value=TextSubstitution(
            text="/dev/v4l/by-path/platform-xhci-hcd.2.auto-usb-0:1.2:1.0-video-index0")
    )
    cam0_fps_launch_arg = DeclareLaunchArgument(
        "cam0_fps", default_value=TextSubstitution(text="25")
    )
    cam1_fps_launch_arg = DeclareLaunchArgument(
        "cam1_fps", default_value=TextSubstitution(text="30")
    )
    sync_tolerance_ms_launch_arg = DeclareLaunchArgument(
        "dnn_example_sync_tolerance_ms",
        default_value=TextSubstitution(text="25")
    )
    websocket_port_launch_arg = DeclareLaunchArgument(
        "websocket_port",
        default_value=TextSubstitution(text="8080")
    )

    # 启动零拷贝环境
    shared_mem_node = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('hobot_shm'),
                'launch/hobot_shm.launch.py'))
    )

    # 双路USB相机
    cam_node_0 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('hobot_usb_cam'),
                'launch/hobot_usb_cam.launch.py')),
        launch_arguments={
            'usb_frame_id': '0',
            'usb_source_id': '0',
            'usb_framerate': LaunchConfiguration('cam0_fps'),
            'usb_image_width': '640',
            'usb_image_height': '480',
            'usb_video_device': LaunchConfiguration('cam0_device'),
            'usb_image_topic': '/image_0',
            'usb_camera_info_topic': '/camera_info_0'
        }.items()
    )

    cam_node_1 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('hobot_usb_cam'),
                'launch/hobot_usb_cam.launch.py')),
        launch_arguments={
            'usb_frame_id': '1',
            'usb_source_id': '1',
            'usb_framerate': LaunchConfiguration('cam1_fps'),
            'usb_image_width': '640',
            'usb_image_height': '480',
            'usb_video_device': LaunchConfiguration('cam1_device'),
            'usb_image_topic': '/image_1',
            'usb_camera_info_topic': '/camera_info_1'
        }.items()
    )

    # 双路codec解码到shared_mem（独立通道）
    nv12_codec_node_0 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('hobot_codec'),
                'launch/hobot_codec_decode.launch.py')),
        launch_arguments={
            'codec_channel': '0',
            'codec_in_mode': 'ros',
            'codec_input_framerate': LaunchConfiguration('cam0_fps'),
            'codec_output_framerate': '15',
            'codec_out_mode': 'shared_mem',
            'codec_sub_topic': '/image_0',
            'codec_pub_topic': '/hbmem_img_0'
        }.items()
    )

    nv12_codec_node_1 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('hobot_codec'),
                'launch/hobot_codec_decode.launch.py')),
        launch_arguments={
            'codec_channel': '1',
            'codec_in_mode': 'ros',
            'codec_input_framerate': LaunchConfiguration('cam1_fps'),
            'codec_output_framerate': '15',
            'codec_out_mode': 'shared_mem',
            'codec_sub_topic': '/image_1',
            'codec_pub_topic': '/hbmem_img_1'
        }.items()
    )

    # DNN batch=2（双路同步）
    dnn_node_example_node = Node(
        package='dnn_node_example',
        executable='example',
        output='screen',
        parameters=[
            {"config_file": LaunchConfiguration('dnn_example_config_file')},
            {"dump_render_img": LaunchConfiguration('dnn_example_dump_render_img')},
            {"feed_type": 1},
            {"is_shared_mem_sub": 1},
            {"batch_size": 2},
            {"enable_batch_sync": 1},
            {"sync_source0_index": 0},
            {"sync_source1_index": 1},
            {"sync_tolerance_ms": LaunchConfiguration('dnn_example_sync_tolerance_ms')},
            {"sharedmem_img_topic_name_0": '/hbmem_img_0'},
            {"sharedmem_img_topic_name_1": '/hbmem_img_1'},
            {"msg_pub_topic_name": LaunchConfiguration('dnn_example_msg_pub_topic_name')},
            {"msg_pub_topic_name_0": LaunchConfiguration('dnn_example_msg_pub_topic_name')},
            {"msg_pub_topic_name_1": [LaunchConfiguration('dnn_example_msg_pub_topic_name'), '_1']}
        ],
        arguments=['--ros-args', '--log-level', 'warn']
    )

    # web展示（默认展示第0路图像和第0路智能结果）
    web_node = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('websocket'),
                'launch/websocket.launch.py')),
        launch_arguments={
            'websocket_image_topic': '/image_0',
            'websocket_image_type': 'mjpeg',
            'websocket_smart_topic': LaunchConfiguration('dnn_example_msg_pub_topic_name'),
            'websocket_server_port': LaunchConfiguration('websocket_port')
        }.items()
    )

    return LaunchDescription([
        config_file_launch_arg,
        dump_render_launch_arg,
        msg_pub_topic_name_launch_arg,
        cam0_device_launch_arg,
        cam1_device_launch_arg,
        cam0_fps_launch_arg,
        cam1_fps_launch_arg,
        sync_tolerance_ms_launch_arg,
        websocket_port_launch_arg,
        shared_mem_node,
        cam_node_0,
        cam_node_1,
        nv12_codec_node_0,
        nv12_codec_node_1,
        dnn_node_example_node,
        web_node
    ])
