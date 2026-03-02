#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

import cv2
from pupil_apriltags import Detector


class AprilTagFollower(Node):

    def __init__(self):
        super().__init__('apriltag_follower')

        # ===== Camera =====
        self.W = 640
        self.H = 480

        self.cap = cv2.VideoCapture(1)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.H)

        # ===== Publisher =====
        self.cmd_pub = self.create_publisher(
            Twist,
            '/teelek/cmd_move',
            10
        )

        # ===== Detector =====
        self.detector = Detector(
            families="tagStandard52h13",
            nthreads=2,
            quad_decimate=2.0
        )

        # ===== Control Gain =====
        self.Kp = 1.5
        self.deadband = 0.05

        self.timer = self.create_timer(0.05, self.update)

        self.get_logger().info("AprilTag Centering with Debug View Started")

    def update(self):

        ret, frame = self.cap.read()
        if not ret:
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detections = self.detector.detect(gray)

        cmd = Twist()
        error = 0.0

        # วาดเส้นกลางจอ
        center_x = int(self.W / 2)
        cv2.line(frame, (center_x, 0), (center_x, self.H), (255, 255, 0), 2)

        if len(detections) > 0:

            tag = detections[0]
            cx = tag.center[0]

            # วาดกรอบ tag
            pts = tag.corners.astype(int)
            cv2.polylines(frame, [pts], True, (0, 255, 0), 2)

            # วาดจุดกลาง tag
            cv2.circle(frame, (int(cx), int(self.H/2)), 6, (0, 0, 255), -1)

            # ===== Normalize error =====
            error = (cx - center_x) / center_x

            if abs(error) < self.deadband:
                error = 0.0

            angular = self.Kp * error
            angular = max(-1.0, min(1.0, angular))

            cmd.angular.z = angular

        else:
            cmd.angular.z = 0.0

        # ===== แสดงค่า error บนภาพ =====
        text = f"Error: {error:.3f}"
        cv2.putText(frame, text,
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 255),
                    2)

        # แสดงค่า angular
        text2 = f"Angular Z: {cmd.angular.z:.3f}"
        cv2.putText(frame, text2,
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 255),
                    2)

        # แสดงภาพ
        frame = cv2.resize(frame, (640, 480))
        cv2.imshow("AprilTag Centering", frame)
        cv2.waitKey(1)

        self.cmd_pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = AprilTagFollower()
    rclpy.spin(node)

    node.cap.release()
    cv2.destroyAllWindows()
    node.destroy_node()
    rclpy.shutdown()


if _name_ == '_main_':
    main()