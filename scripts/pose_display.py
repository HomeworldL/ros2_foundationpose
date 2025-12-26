#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterEvent, ParameterDescriptor, ParameterValue
from sensor_msgs.msg import Image, CameraInfo, CompressedImage
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
import cv2
from message_filters import Subscriber, TimeSynchronizer

# from vision_msgs
from ros2_foundationpose.estimater import (
    FoundationPose,
    ScorePredictor,
    PoseRefinePredictor,
)
from ros2_foundationpose.Utils import dr, draw_posed_3d_box, draw_xyz_axis
from ros2_foundationpose.detector import GroundingDINOModel
import trimesh
import numpy as np
import time
from scipy.spatial.transform import Rotation
from supervision import Detections
import logging


class PoseDisplayNode(Node):
    def __init__(self):
        super().__init__("pose_display_node")

        self.declare_parameters(
            namespace="",
            parameters=[
                ("obj_file", "", ParameterDescriptor(description="Object .obj file")),
                ("obj_name", "", ParameterDescriptor(description="Object name")),
                ("image_topic", "", ParameterDescriptor(description="Image topic")),
                ("depth_topic", "", ParameterDescriptor(description="Depth topic")),
                (
                    "camera_info_topic",
                    "",
                    ParameterDescriptor(description="Camera info topic"),
                ),
            ],
        )
        self.obj_file = self.get_parameter("obj_file").value
        self.get_logger().info(self.obj_file)
        self.obj_name = self.get_parameter("obj_name").value
        self.mesh = trimesh.load(self.obj_file)
        # self.mesh.apply_scale(1/1000)
        # self.mesh.show()
        self.to_origin, self.extents = trimesh.bounds.oriented_bounds(self.mesh)
        self.bbox = np.stack([-self.extents / 2, self.extents / 2], axis=0).reshape(
            2, 3
        )

        self.bridge = CvBridge()

        # ROS2
        self.sub_color = Subscriber(
            self,
            CompressedImage,
            self.get_parameter("image_topic").value + "/compressed",
        )
        self.sub_depth = Subscriber(
            self, Image, self.get_parameter("depth_topic").value
        )
        self.ts = TimeSynchronizer([self.sub_color, self.sub_depth], 10)
        self.ts.registerCallback(self.image_callback)
        
        self.camera_info_received = False
        self.sub_camera_info = self.create_subscription(
            CameraInfo,
            self.get_parameter("camera_info_topic").value,
            self.camera_info_callback,
            1,
        )

        self.sub_pose = self.create_subscription(
            PoseStamped, "/foundationpose/" + self.obj_name, self.pose_callback,10
        )
        self.timer_display = self.create_timer(1 / 30, self.display_loop)

        self.rgb = None
        self.bgr = None
        self.depth = None
        self.K = None
        self.pose = None
        self.i = 0

    def image_callback(self, color_msg: CompressedImage, depth_msg: Image):
        tstart = time.time()
        if self.camera_info_received is False:
            return

        # rgb = self.bridge.imgmsg_to_cv2(color_msg)  # rgb
        self.rgb = self.bridge.compressed_imgmsg_to_cv2(
            color_msg, desired_encoding="rgb8"
        )
        self.bgr = cv2.cvtColor(self.rgb, cv2.COLOR_RGB2BGR)
        self.depth = self.bridge.imgmsg_to_cv2(depth_msg) / 1000
        self.get_logger().debug(f"time for convert: {time.time()-tstart}")
        
    def pose_callback(self, msg: PoseStamped):
        self.pose = msg.pose
        
        # homogeneous
        pose = np.zeros((4, 4))
        pose[0,3] = msg.pose.position.x
        pose[1,3] = msg.pose.position.y
        pose[2,3] = msg.pose.position.z
        quat = [msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w]
        pose[:3, :3] = Rotation.from_quat(quat).as_matrix()
        self.pose = pose

    def camera_info_callback(self, info: CameraInfo):
        if self.camera_info_received is False:
            self.camera_d = info.d
            self.camera_height = info.height
            self.camera_width = info.width
            self.camera_k = info.k
            self.K = np.array(self.camera_k).reshape((3, 3))
            self.camera_r = info.r
            self.binning_x = info.binning_x
            self.get_logger().info(
                f"camera_height: {self.camera_height}\n \
                    camera_width: {self.camera_width}\n \
                    camera_d: {self.camera_d}\n \
                    camera_k: {self.camera_k}\n \
                    camera_r: {self.camera_r}\n \
                    binning_x: {self.binning_x}\n"
            )
            self.camera_info_received = True
        else:
            return

    def display_loop(self):
        
        if self.rgb is None or self.pose is None or self.K is None:
            return

        tstart = time.time()
        cv2.imshow("Image", self.bgr)
        # cv2.imshow(self.obj_name, self.visualization)
        center_pose = self.pose @ np.linalg.inv(self.to_origin)
        vis = draw_posed_3d_box(
            self.K, img=self.rgb.copy(), ob_in_cam=center_pose, bbox=self.bbox
        )
        vis = draw_xyz_axis(
            vis,
            ob_in_cam=self.pose,
            scale=0.1,
            K=self.K,
            thickness=3,
            transparency=0,
            is_input_rgb=True,
        )
        cv2.imshow("pose", vis[..., ::-1])

        if cv2.waitKey(1) & 0xFF == ord("q"):
            exit(1)
        self.get_logger().debug(f"time for show: {time.time()-tstart}")


def main(args=None):
    rclpy.init(args=args)
    node = PoseDisplayNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Interrupted by user, shutting down...")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
