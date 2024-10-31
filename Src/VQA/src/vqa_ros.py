#!/usr/bin/env python

from PIL import Image
import torch

import rospy
# from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import String
import cv2
import os

from vqa_unit_test import VQAInference

import rospy
from vqa.srv import GetDescription, GetDescriptionResponse
from detection.srv import GetImage

class VQA:

    def __init__(self, model_path, pub=None):
        self.publisher = pub
        # self.bridge = CvBridge()
        self.color_image = None
        self.img_counter = 0
        self.save_dir = os.path.join(os.getcwd(), 'image_temp')
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        # load and initialize models related to ClipClap
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        config = {'exp_name': 'vilt', 'seed': 0, 'datasets': ['coco', 'vg', 'sbu', 'gcc'],
                  'loss_names': {"itm": 0, "mlm": 0, "mpp": 0, "vqa": 1, "imgcls": 0, "nlvr2": 0, "irtr": 0, "arc": 0},
                  'batch_size': 4096,
                  'train_transform_keys': ['pixelbert'], 'val_transform_keys': ['pixelbert'], 'image_size': 384,
                  'max_image_len': -1,
                  'patch_size': 32, 'draw_false_image': 1, 'image_only': False, 'vqav2_label_size': 3129,
                  'max_text_len': 40,
                  'tokenizer': 'bert-base-uncased', 'vocab_size': 30522, 'whole_word_masking': False, 'mlm_prob': 0.15,
                  'draw_false_text': 0,
                  'vit': 'vit_base_patch32_384', 'hidden_size': 768, 'num_heads': 12, 'num_layers': 12, 'mlp_ratio': 4,
                  'drop_rate': 0.1,
                  'optim_type': 'adamw', 'learning_rate': 1e-06, 'weight_decay': 0.01, 'decay_power': 1,
                  'max_epoch': 100, 'max_steps': 25000,
                  'warmup_steps': 2500, 'end_lr': 0, 'lr_mult': 1, 'get_recall_metric': False, 'resume_from': None,
                  'fast_dev_run': False,
                  'val_check_interval': 1.0, 'test_only': True, 'data_root': '', 'log_dir': 'result',
                  'per_gpu_batchsize': 0, 'num_gpus': 1,
                  'num_nodes': 1,
                  'load_path': model_path,
                  'num_workers': 8, 'precision': 16}

        self.vqa_module = VQAInference(config)

        self.question = None

    # we MUST set question before calling vqa_callback!
    def set_question(self, question):
        self.question = question

    def image_service(path):

        rospy.loginfo('waiting for get image service...')
        rospy.wait_for_service('imaging_server')

        send = rospy.ServiceProxy('imaging_server', GetImage)

        image_name = send(path)

        return image_name

    def vqa_callback(self):
        # # convert ros image to PIL image
        # self.color_image_timestamp = img_msg.header.stamp.secs
        # self.color_image = self.bridge.compressed_imgmsg_to_cv2(img_msg, desired_encoding='bgr8')

        # # save the image for debugging
        # filename = str(self.img_counter) + ".png"
        # cv2.imwrite(os.path.join(self.save_dir, filename), self.color_image)

        image_file_path = '/home/mustar/test_ws/src/vqa/scripts/Output_Pics'
        image_name = self.image_service(image_file_path)
        image_name = image_name.name
        image_path = os.join(image_file_path, image_name)

        # color_coverted = cv2.cvtColor(self.color_image, cv2.COLOR_BGR2RGB)
        # pil_image = PIL.Image.fromarray(color_coverted).convert("RGB")

        pil_image = Image.open(image_path).convert("RGB")

        # generate answers
        if self.question is None:
            print('MUST set question before calling vqa_callback!')
            return
        answer = self.vqa_module.infer(pil_image, self.question)
        print("Predicted answer:", answer)

        if self.publisher is not None:
            self.publisher.publish(answer)

        return answer


def main(question):
    try:
        rospy.init_node('image_captioning', log_level=rospy.INFO, disable_signals=True)
        rospy.loginfo('image_captioning node started')

        # publisher for the caption text
        caption_pub = rospy.Publisher('/image_caption_text', String, queue_size=2)

        vqa = VQA(pub=caption_pub)
        vqa.question = question #"How many cats are there?"

        # for realsense:
        # RGB image
        #   /camera/color/image_raw
        # depth image
        #   /camera/depth/image_raw
        # rospy.Subscriber('/camera/rgb/image_raw', Image, converter.color_image_callback, queue_size=1)

        while not rospy.is_shutdown():
            # todo: change to rospy.Subscriber('/camera/color/image_raw/compressed', CompressedImage)!
            # data = rospy.wait_for_message('/camera/rgb/image_raw/compressed', CompressedImage, timeout=5)
            # answer = vqa.vqa_callback(data)
            answer = vqa.vqa_callback()
            rospy.sleep(5)
        
        return answer

    except rospy.ROSInterruptException:
        exit()


def get(request):

    rospy.loginfo('Received request...')
    response = main(request.question)
    rospy.loginfo('Request finished...')
        
    return GetDescriptionResponse(response)



if __name__=="__main__":

    rospy.init_node('vilt_server')
    rospy.loginfo('ViLT Server Initiated')

    try:
        
        service = rospy.Service('description_service', GetDescription, get)
        rospy.spin()

    except rospy.ROSInterruptException:
        pass

