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


class FoundationPoseNode(Node):
    def __init__(self):
        super().__init__("foundationpose_node")

        self.declare_parameters(
            namespace="",
            parameters=[
                ("device", "", ParameterDescriptor(description="Device")),
                ("obj_file", "", ParameterDescriptor(description="Object .obj file")),
                ("obj_name", "", ParameterDescriptor(description="Object name")),
                ("image_topic", "", ParameterDescriptor(description="Image topic")),
                ("depth_topic", "", ParameterDescriptor(description="Depth topic")),
                (
                    "camera_info_topic",
                    "",
                    ParameterDescriptor(description="Camera info topic"),
                ),
                (
                    "est_refine_iter",
                    5,
                    ParameterDescriptor(description="Estimate refinement iterations"),
                ),
                (
                    "track_refine_iter",
                    2,
                    ParameterDescriptor(description="Track refinement iterations"),
                ),
                (
                    "log_level",
                    "warning",
                    ParameterDescriptor(description="Debug level"),
                ),
                ("display", True, ParameterDescriptor(description="Display")),
            ],
        )

        self.log_level = self.get_parameter("log_level").value
        if self.log_level == "debug":
            logging.getLogger().setLevel(logging.DEBUG)
        elif self.log_level == "info":
            logging.getLogger().setLevel(logging.INFO)
        elif self.log_level == "warning":
            logging.getLogger().setLevel(logging.WARNING)
        elif self.log_level == "error":
            logging.getLogger().setLevel(logging.ERROR)
        self.display = self.get_parameter("display").value
        self.device = self.get_parameter("device").value
        self.obj_file = self.get_parameter("obj_file").value
        self.obj_name = self.get_parameter("obj_name").value
        self.est_refine_iter = self.get_parameter("est_refine_iter").value
        self.track_refine_iter = self.get_parameter("track_refine_iter").value
        self.mesh = trimesh.load(self.obj_file)
        # self.mesh.apply_scale(1/1000)
        # self.mesh.show()
        self.to_origin, self.extents = trimesh.bounds.oriented_bounds(self.mesh)
        self.bbox = np.stack([-self.extents / 2, self.extents / 2], axis=0).reshape(
            2, 3
        )

        # set_logging_format()
        # set_seed(0)
        debug_dir = "./debug"
        scorer = ScorePredictor()
        refiner = PoseRefinePredictor()
        glctx = dr.RasterizeCudaContext()
        self.est = FoundationPose(
            model_pts=self.mesh.vertices,
            model_normals=self.mesh.vertex_normals,
            mesh=self.mesh,
            scorer=scorer,
            refiner=refiner,
            debug_dir=debug_dir,
            debug=0,
            glctx=glctx,
        )

        self.get_logger().info("Load FoundationPose")
        self.detector = GroundingDINOModel(self.device)
        self.get_logger().info("Load GroundingDINO Detector")

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
        self.pub_pose = self.create_publisher(
            PoseStamped, "/foundationpose/" + self.obj_name, 10
        )
        self.timer_estimate = self.create_timer(1 / 30, self.estimate_loop)
        self.timer_display = self.create_timer(1 / 30, self.display_loop)

        self.rgb = None
        self.bgr = None
        self.depth = None
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

    def estimate_loop(self):
        if self.rgb is None or self.depth is None:
            return

        tstart = time.time()
        if self.i == 0:
            grounding_caption = self.obj_name
            detections, labels = self.detector.run_grounding_caption(
                image=self.bgr.copy(),
                caption=grounding_caption,
            )

            if len(detections) > 0:
                # 获取置信度最大的检测
                max_conf_idx = np.argmax(detections.confidence)
                single_detection_xyxy = detections.xyxy[max_conf_idx : max_conf_idx + 1]
                single_detection_conf = detections.confidence[
                    max_conf_idx : max_conf_idx + 1
                ]
                single_detection_class_id = detections.class_id[
                    max_conf_idx : max_conf_idx + 1
                ]
                single_detection = Detections(
                    xyxy=single_detection_xyxy,
                    confidence=single_detection_conf,
                    class_id=single_detection_class_id,
                )
                single_label = [labels[max_conf_idx]]

                self.visualization = self.detector.get_detection_image_labels(
                    self.bgr.copy(), single_detection, single_label
                )

            mask = np.zeros_like(self.depth)
            mask = np.zeros(self.depth.shape[:2], dtype=np.uint8)
            for det in single_detection.xyxy:
                xmin, ymin, xmax, ymax = det[:4].astype(int)
                cv2.rectangle(mask, (xmin, ymin), (xmax, ymax), (255), -1)
            # cv2.imshow("Mask", mask)
            mask = mask.astype(bool)
            # print(f"type(rgb): {rgb.dtype}, type(depth): {depth.dtype}, type(K): {K.dtype}, type(mask): {mask.dtype}")
            pose = self.est.register(
                K=self.K,
                rgb=self.rgb.copy(),
                depth=self.depth.copy().astype(np.float64),
                ob_mask=mask,
                iteration=self.est_refine_iter,
            )

        else:
            pose = self.est.track_one(
                rgb=self.rgb.copy(),
                depth=self.depth.copy(),
                K=self.K,
                iteration=self.track_refine_iter,
            )

        self.get_logger().debug(f"time for est: {time.time()-tstart}")
        pose.astype("float")

        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.pose.position.x = float(pose[0, 3])
        pose_msg.pose.position.y = float(pose[1, 3])
        pose_msg.pose.position.z = float(pose[2, 3])
        r = Rotation.from_matrix(pose[:3, :3])
        quat = r.as_quat()
        pose_msg.pose.orientation.x = quat[0]
        pose_msg.pose.orientation.y = quat[1]
        pose_msg.pose.orientation.z = quat[2]
        pose_msg.pose.orientation.w = quat[3]
        self.pub_pose.publish(pose_msg)
        self.pose = pose

        self.i += 1

    def display_loop(self):

        if self.display:
            if self.rgb is None or self.pose is None or self.K is None:
                return
            
            tstart = time.time()
            cv2.imshow("Image", self.bgr)
            cv2.imshow(self.obj_name, self.visualization)
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
    node = FoundationPoseNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Interrupted by user, shutting down...")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
