#!/usr/bin/env python

import PIL
import torch

import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import String
import cv2
import os


# path = '/home/mustar/test_ws/src/detection/GroundingDINO/cans.mp4'
# bridge = CvBridge()
# img = bridge.imgmsg_to_cv2(img_msg, "bgr8")



def vqa_callback(img_msg):
        # convert ros image to PIL image
        # color_image_timestamp = img_msg.header.stamp.secs
        # color_image = CvBridge().compressed_imgmsg_to_cv2(img_msg, desired_encoding='bgr8')

        # # save the image for debugging
        # filename = str(self.img_counter) + ".png"
        # cv2.imwrite(os.path.join(self.save_dir, filename), self.color_image)

        # print(img_msg)
        # bridge = CvBridge()
        img_msg.show()
        # img = bridge.imgmsg_to_cv2(img_msg, "bgr8")
        
        # Display the image
        # cv2.imshow('live vid', img)
        # cv2.imshow('live vid', img_msg)

        # color_coverted = cv2.cvtColor(self.color_image, cv2.COLOR_BGR2RGB)
        # pil_image = PIL.Image.fromarray(color_coverted).convert("RGB")

        # # generate answers
        # if self.question is None:
        #     print('MUST set question before calling vqa_callback!')
        #     return
        # answer = self.vqa_module.infer(pil_image, self.question)
        # print("Predicted answer:", answer)

        # if self.publisher is not None:
        #     self.publisher.publish(answer)




if __name__ == '__main__':
    try:
        rospy.init_node('image_captioning', log_level=rospy.INFO, disable_signals=True)
        rospy.loginfo('image_captioning node started')

        # # publisher for the caption text
        # caption_pub = rospy.Publisher('/image_caption_text', String, queue_size=2)

        # vqa = VQA(pub=caption_pub)
        # vqa.question = "How many cats are there?"

        # for realsense:
        # RGB image
        #   /camera/color/image_raw
        # depth image
        #   /camera/depth/image_raw
        # rospy.Subscriber('/camera/rgb/image_raw', Image, converter.color_image_callback, queue_size=1)
        print("hello")
        while not rospy.is_shutdown():
            # todo: change to rospy.Subscriber('/camera/color/image_raw/compressed', CompressedImage)!
            data = rospy.wait_for_message('/camera/rgb/image_raw', Image, timeout=5)
            vqa_callback(data)
            rospy.sleep(5)

    except rospy.ROSInterruptException:
        exit()



# # load image
# image_pil = Image.open('/home/mustar/test_ws/src/detection/GroundingDINO/1.jpg').convert("RGB")  # load image
# image_pil.show()

