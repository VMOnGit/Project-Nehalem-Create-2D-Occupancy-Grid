#!/usr/bin/env python3
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from nav_msgs.msg import OccupancyGrid

class ImageListner(Node):
        images = []
        def __init__(self):
            super().__init__('Image_listener')
            self.subscription=self.create_subscription(
                Image,
                '/overhead_camera/overhead_camera1/image_raw',
                self.listener_callback1,
                10
            )
            
            self.subscription=self.create_subscription(
                Image,
                '/overhead_camera/overhead_camera2/image_raw',
                self.listener_callback2,
                10
            )
            self.subscription=self.create_subscription(
                Image,
                '/overhead_camera/overhead_camera3/image_raw',
                self.listener_callback3,
                10
            )
            self.subscription=self.create_subscription(
                Image,
                '/overhead_camera/overhead_camera4/image_raw',
                self.listener_callback4,
                10
            )
            self.subscription=self.create_subscription(
                Image,
                '/stitcher',
                self.create_occupancy_grid,
                10
            )
            self.publisher_= self.create_publisher(OccupancyGrid,'occupancy_grid',10)
            self.publisher = self.create_publisher(Image, '/stitched',10)
            self.bridge = CvBridge()

        



        def listener_callback1(self, msg):
            
            self.get_logger().info('Receiving Image')
            cv_image1 = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            
            self.images.append(cv_image1)
            
           


        def listener_callback2(self, msg):
            
            self.get_logger().info('Receiving Image')
            cv_image2 = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
           
            self.images.append(cv_image2)
            


        def listener_callback3(self, msg):
            self.get_logger().info('Receiving Image')
            cv_image3 = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            
            self.images.append(cv_image3)
           

        def listener_callback4(self, msg):

            self.get_logger().info('Receiving Image')
            cv_image4 = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            
            self.images.append(cv_image4)
            self.image_stitch()

    
        

        def image_stitch(self):
            sift = cv2.SIFT_create()
            images8bit = []
            
            # Convert images to grayscale and display if needed
            for img in self.images:
                imageGS= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                images8bit.append(imageGS)

            # Find the keypoints and descriptors with SIFT
            keypoints_descriptors = [sift.detectAndCompute(img, None) for img in images8bit]

            # Initialize a matcher
            bf = cv2.BFMatcher()

            all_good_matches = []
            all_homographies = []

            # Find the matches between the images
            for i in range(len(self.images) - 1):
                kp1, des1 = keypoints_descriptors[i]
                kp2, des2 = keypoints_descriptors[i+1]
                matches = bf.knnMatch(des1, des2, k=4)
                
                # Apply ratio test to find good matches
                good_matches = []
                for m_n in matches:
                    if len(m_n) >= 4:
                        m1, m2, m3, m4 = m_n[0], m_n[1], m_n[2], m_n[3]
                        if (m1.distance < 0.75 * m2.distance and
                            m1.distance < 0.75 * m3.distance and
                            m1.distance < 0.75 * m4.distance):
                            good_matches.append(m1)
                
                all_good_matches.append(good_matches)
                
                # Extract the matched keypoints
                src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                # Find the homography matrix
                H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                all_homographies.append(H)

            # Determine the size of the new result canvas
            height, width = self.images[0].shape[:2]
            max_width = width
            total_height = height
            
            for i in range(1, len(self.images)):
                H = all_homographies[i-1]
                
                # Project the corner points to find the new dimensions
                corners = np.array([[0, 0], [0, height], [width, height], [width, 0]], dtype='float32').reshape(-1, 1, 2)
                projected_corners = cv2.perspectiveTransform(corners, H)
                
                # Update the maximum width and total height
                max_width = max(max_width, int(projected_corners[2][0][0]))
                total_height = max(total_height, int(projected_corners[2][0][1]))

            # Create a blank canvas for the stitched image
            result = np.zeros((total_height, max_width, 3), dtype=np.uint8)

            # Warp and stitch images
            result[0:height, 0:width] = self.images[0]

            for i in range(1, len(self.images)):
                H = all_homographies[i-1]
                warp_result = cv2.warpPerspective(self.images[i], H, (max_width, total_height))
                result = np.maximum(result, warp_result)
            cv2.imshow("Stitched", result)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            cv2.waitKey(1)
            self.publisher.publish(result)
            




        def create_occupancy_grid(self, msg):
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            
            _, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
            cv2.imshow("bin", gray_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            cv2.waitKey(1)
            occupancy_grid = OccupancyGrid()
            
            occupancy_grid.header.stamp =ImageListner.get_clock(self).now().to_msg()
            occupancy_grid.header.frame_id = "map"
            
            occupancy_grid.info.resolution = 0.05
            occupancy_grid.info.width = binary_image.shape[1]
            occupancy_grid.info.height = binary_image.shape[0]
            occupancy_grid.info.origin.position.x = 0.0
            occupancy_grid.info.origin.position.y = 0.0
            occupancy_grid.info.origin.position.z = 0.0
            occupancy_grid.info.origin.orientation.w = 1.0
            
            data = []
            for i in range(binary_image.shape[0]):
                for j in range(binary_image.shape[1]):
                    if binary_image[i, j] == 255:
                        data.append(0)
                    else:
                        data.append(100)
            occupancy_grid.data = data
            self.publischer_.publisch(occupancy_grid)
        
    
       
def main(args=None):
    rclpy.init(args=args)
    node = ImageListner()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

main()