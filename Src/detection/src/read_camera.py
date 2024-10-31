#!/usr/bin/env python

import os
import rospy
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from detection.srv import GetImage, GetImageResponse

path = image_name = ''


def image_callback(data):

    global image_name

    bridge = CvBridge()

    try:
        # Convert ROS Image message to OpenCV image
        rospy.loginfo('start taking image')
        cv_image = bridge.imgmsg_to_cv2(data, "bgr8")
        rospy.loginfo('image taken')

        # rospy.signal_shutdown("Received a message, shutting down...")

        # Generate filename based on current ROS time
        current_time = rospy.Time.now().to_sec()
        image_name = "astra_image_{}.jpg".format(current_time)
        image_path = os.path.join(path, image_name)

        cv2.imwrite(image_path, cv_image)
        rospy.loginfo("Image saved at: {}".format(image_path))

    except Exception as e:
        print(e)

def get(request):
    global path
    path = request.path
    rospy.loginfo('start subscribing')
    try:
        image_sub = rospy.Subscriber('/camera/rgb/image_raw', Image, image_callback)
    
        # Wait for a single message
        rospy.wait_for_message("/camera/rgb/image_raw", Image)
        
        # Unregister the subscriber after receiving the first message
        image_sub.unregister()

        rospy.loginfo('image saved in {}'.format(image_name))
        
        return GetImageResponse(image_name)

    except rospy.ROSInterruptException:
        pass



if __name__=="__main__":

    rospy.init_node('imaging_server')
    rospy.loginfo('Imaging Server Initiated')

    try:
        
        service = rospy.Service('imaging_server', GetImage, get)
        rospy.spin()

    except rospy.ROSInterruptException:
        pass

