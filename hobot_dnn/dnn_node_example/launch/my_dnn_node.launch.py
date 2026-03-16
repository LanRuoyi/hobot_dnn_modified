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
from launch.substitutions import PythonExpression
from launch.conditions import IfCondition, UnlessCondition
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

    # args that can be set from the command line or a default will be used
    config_file_launch_arg = DeclareLaunchArgument(
        "dnn_example_config_file", default_value=TextSubstitution(text="config/yolo11workconfig.json")
    )
    dump_render_launch_arg = DeclareLaunchArgument(
        "dnn_example_dump_render_img", default_value=TextSubstitution(text="0")
    )
    image_width_launch_arg = DeclareLaunchArgument(
        "dnn_example_image_width", default_value=TextSubstitution(text="960")
    )
    image_height_launch_arg = DeclareLaunchArgument(
        "dnn_example_image_height", default_value=TextSubstitution(text="544")
    )
    msg_pub_topic_name_launch_arg = DeclareLaunchArgument(
        "dnn_example_msg_pub_topic_name", default_value=TextSubstitution(text="hobot_dnn_detection")
    )
    enable_batch2_launch_arg = DeclareLaunchArgument(
        "enable_batch2", default_value=TextSubstitution(text="False")
    )
    dnn_batch_size_launch_arg = DeclareLaunchArgument(
        "dnn_example_batch_size", default_value=TextSubstitution(text="1")
    )
    dnn_enable_batch_sync_launch_arg = DeclareLaunchArgument(
        "dnn_example_enable_batch_sync", default_value=TextSubstitution(text="False")
    )
    sync_source0_idx_launch_arg = DeclareLaunchArgument(
        "dnn_example_sync_source0_idx", default_value=TextSubstitution(text="0")
    )
    sync_source1_idx_launch_arg = DeclareLaunchArgument(
        "dnn_example_sync_source1_idx", default_value=TextSubstitution(text="1")
    )
    sync_tolerance_ms_launch_arg = DeclareLaunchArgument(
        "dnn_example_sync_tolerance_ms", default_value=TextSubstitution(text="25")
    )

    cam0_device_launch_arg = DeclareLaunchArgument(
        "cam0_device",
        default_value=TextSubstitution(text="/dev/v4l/by-path/platform-xhci-hcd.2.auto-usb-0:1.1:1.0-video-index0")
    )
    cam1_device_launch_arg = DeclareLaunchArgument(
        "cam1_device",
        default_value=TextSubstitution(text="/dev/v4l/by-path/platform-xhci-hcd.2.auto-usb-0:1.2:1.0-video-index0")
    )
    cam0_fps_launch_arg = DeclareLaunchArgument(
        "cam0_fps", default_value=TextSubstitution(text="25")
    )
    cam1_fps_launch_arg = DeclareLaunchArgument(
        "cam1_fps", default_value=TextSubstitution(text="30")
    )

    websocket_dual_display_launch_arg = DeclareLaunchArgument(
        "websocket_dual_display", default_value=TextSubstitution(text="False")
    )
    websocket_port_0_launch_arg = DeclareLaunchArgument(
        "websocket_port_0", default_value=TextSubstitution(text="8080")
    )
    websocket_port_1_launch_arg = DeclareLaunchArgument(
        "websocket_port_1", default_value=TextSubstitution(text="8082")
    )

    camera_type = os.getenv('CAM_TYPE')
    print("camera_type is ", camera_type)
    
    cam_node_1_legacy = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('hobot_usb_cam'),
                'launch/hobot_usb_cam.launch.py')),
        condition=UnlessCondition(LaunchConfiguration('enable_batch2')),
        launch_arguments={
            'usb_frame_id': '0',
            'usb_source_id': '0',
            'usb_framerate': LaunchConfiguration('cam0_fps'),
            'usb_image_width': '640',
            'usb_image_height': '480',
            'usb_video_device': LaunchConfiguration('cam0_device'),
            'usb_image_topic': '/image_1',
            'usb_camera_info_topic': '/camera_info_1'
        }.items()
    )
    cam_node_2_legacy = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('hobot_usb_cam'),
                'launch/hobot_usb_cam.launch.py')),
        condition=UnlessCondition(LaunchConfiguration('enable_batch2')),
        launch_arguments={
            'usb_frame_id': '1',
            'usb_source_id': '1',
            'usb_framerate': LaunchConfiguration('cam1_fps'),
            'usb_image_width': '640',
            'usb_image_height': '480',
            'usb_video_device': LaunchConfiguration('cam1_device'),
            'usb_image_topic': '/image_2',
            'usb_camera_info_topic': '/camera_info_2'
        }.items()
    )
    cam_node_1_batch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('hobot_usb_cam'),
                'launch/hobot_usb_cam.launch.py')),
        condition=IfCondition(LaunchConfiguration('enable_batch2')),
        launch_arguments={
            'usb_frame_id': '0',
            'usb_source_id': '0',
            'usb_framerate': LaunchConfiguration('cam0_fps'),
            'usb_image_width': '640',
            'usb_image_height': '480',
            'usb_video_device': LaunchConfiguration('cam0_device'),
            'usb_image_topic': '/image_batch_mjpeg',
            'usb_camera_info_topic': '/camera_info_1'
        }.items()
    )
    cam_node_2_batch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('hobot_usb_cam'),
                'launch/hobot_usb_cam.launch.py')),
        condition=IfCondition(LaunchConfiguration('enable_batch2')),
        launch_arguments={
            'usb_frame_id': '1',
            'usb_source_id': '1',
            'usb_framerate': LaunchConfiguration('cam1_fps'),
            'usb_image_width': '640',
            'usb_image_height': '480',
            'usb_video_device': LaunchConfiguration('cam1_device'),
            'usb_image_topic': '/image_batch_mjpeg',
            'usb_camera_info_topic': '/camera_info_2'
        }.items()
    )
    print("using usb cam")

    # nv12图片解码&发布pkg
    nv12_codec_node_1 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('hobot_codec'),
                'launch/hobot_codec_decode.launch.py')),
        condition=UnlessCondition(LaunchConfiguration('enable_batch2')),
        launch_arguments={
            'codec_channel': '0',
            'codec_in_mode': 'ros',
            'codec_input_framerate': LaunchConfiguration('cam0_fps'),
            'codec_output_framerate': '15',
            'codec_out_mode': 'shared_mem',
            'codec_keep_input_index': 'True',
            'codec_parse_input_frame_id_as_index': 'True',
            'codec_sub_topic': '/image_1',
            'codec_pub_topic': '/hbmem_img'
        }.items()
    )

    nv12_codec_node_2 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('hobot_codec'),
                'launch/hobot_codec_decode.launch.py')),
        condition=UnlessCondition(LaunchConfiguration('enable_batch2')),
        launch_arguments={
            'codec_channel': '1',
            'codec_in_mode': 'ros',
            'codec_input_framerate': LaunchConfiguration('cam1_fps'),
            'codec_output_framerate': '15',
            'codec_out_mode': 'shared_mem',
            'codec_keep_input_index': 'True',
            'codec_parse_input_frame_id_as_index': 'True',
            'codec_sub_topic': '/image_2',
            'codec_pub_topic': '/hbmem_img'
        }.items()
    )
    nv12_codec_node_batch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('hobot_codec'),
                'launch/hobot_codec_decode.launch.py')),
        condition=IfCondition(LaunchConfiguration('enable_batch2')),
        launch_arguments={
            'codec_channel': '0',
            'codec_in_mode': 'ros',
            'codec_input_framerate': '55',
            'codec_output_framerate': '30',
            'codec_out_mode': 'shared_mem',
            'codec_keep_input_index': 'True',
            'codec_parse_input_frame_id_as_index': 'True',
            'codec_sub_topic': '/image_batch_mjpeg',
            'codec_pub_topic': '/hbmem_img'
        }.items()
    )

    # web展示pkg
    web_node_single_legacy = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('websocket'),
                'launch/websocket.launch.py')),
        condition=IfCondition(PythonExpression([
            "not ", LaunchConfiguration('enable_batch2'),
            " and not ", LaunchConfiguration('websocket_dual_display')
        ])),
        launch_arguments={
            'websocket_image_topic': '/image_1',
            'websocket_image_type': 'mjpeg',
            'websocket_smart_topic': LaunchConfiguration("dnn_example_msg_pub_topic_name"),
            'websocket_server_port': LaunchConfiguration('websocket_port_0')
        }.items()
    )
    web_node_single_batch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('websocket'),
                'launch/websocket.launch.py')),
        condition=IfCondition(PythonExpression([
            LaunchConfiguration('enable_batch2'),
            " and not ", LaunchConfiguration('websocket_dual_display')
        ])),
        launch_arguments={
            'websocket_image_topic': '/image_batch_mjpeg',
            'websocket_image_type': 'mjpeg',
            'websocket_smart_topic': LaunchConfiguration("dnn_example_msg_pub_topic_name"),
            'websocket_server_port': LaunchConfiguration('websocket_port_0')
        }.items()
    )
    web_node_cam0 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('websocket'),
                'launch/websocket.launch.py')),
        condition=IfCondition(PythonExpression([
            "not ", LaunchConfiguration('enable_batch2'),
            " and ", LaunchConfiguration('websocket_dual_display')
        ])),
        launch_arguments={
            'websocket_image_topic': '/image_1',
            'websocket_image_type': 'mjpeg',
            'websocket_only_show_image': 'True',
            'websocket_server_port': LaunchConfiguration('websocket_port_0')
        }.items()
    )
    web_node_cam1 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('websocket'),
                'launch/websocket.launch.py')),
        condition=IfCondition(PythonExpression([
            "not ", LaunchConfiguration('enable_batch2'),
            " and ", LaunchConfiguration('websocket_dual_display')
        ])),
        launch_arguments={
            'websocket_image_topic': '/image_2',
            'websocket_image_type': 'mjpeg',
            'websocket_only_show_image': 'True',
            'websocket_server_port': LaunchConfiguration('websocket_port_1')
        }.items()
    )

    # 算法pkg
    dnn_node_example_node = Node(
        package='dnn_node_example',
        executable='example',
        output='screen',
        parameters=[
            {"config_file": LaunchConfiguration('dnn_example_config_file')},
            {"dump_render_img": LaunchConfiguration(
                'dnn_example_dump_render_img')},
            {"feed_type": 1},
            {"is_shared_mem_sub": 1},
            {"batch_size": LaunchConfiguration('dnn_example_batch_size')},
            {"enable_batch_sync": LaunchConfiguration('dnn_example_enable_batch_sync')},
            {"sync_source0_index": LaunchConfiguration('dnn_example_sync_source0_idx')},
            {"sync_source1_index": LaunchConfiguration('dnn_example_sync_source1_idx')},
            {"sync_tolerance_ms": LaunchConfiguration('dnn_example_sync_tolerance_ms')},
            {"msg_pub_topic_name": LaunchConfiguration(
                "dnn_example_msg_pub_topic_name")}
        ],
        arguments=['--ros-args', '--log-level', 'warn']
    )

    shared_mem_node = IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(
                        get_package_share_directory('hobot_shm'),
                        'launch/hobot_shm.launch.py'))
            )
    
    return LaunchDescription([
        # camera_device_arg,
        config_file_launch_arg,
        dump_render_launch_arg,
        image_width_launch_arg,
        image_height_launch_arg,
        msg_pub_topic_name_launch_arg,
        enable_batch2_launch_arg,
        dnn_batch_size_launch_arg,
        dnn_enable_batch_sync_launch_arg,
        sync_source0_idx_launch_arg,
        sync_source1_idx_launch_arg,
        sync_tolerance_ms_launch_arg,
        cam0_device_launch_arg,
        cam1_device_launch_arg,
        cam0_fps_launch_arg,
        cam1_fps_launch_arg,
        websocket_dual_display_launch_arg,
        websocket_port_0_launch_arg,
        websocket_port_1_launch_arg,
        # 启动零拷贝环境配置node
        shared_mem_node,
        # 图片发布pkg
        cam_node_1_legacy,
        cam_node_2_legacy,
        cam_node_1_batch,
        cam_node_2_batch,
        # 图片编解码&发布pkg
        nv12_codec_node_1,
        nv12_codec_node_2,
        nv12_codec_node_batch,
        # 启动example pkg
        dnn_node_example_node,
        # 启动web展示pkg
        web_node_single_legacy,
        web_node_single_batch,
        web_node_cam0,
        web_node_cam1
    ])
