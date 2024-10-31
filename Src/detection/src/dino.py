#!/usr/bin/env python

import os
import sys
import argparse
import subprocess

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from groundingdino.util.vl_utils import create_positive_map_from_span

import rospy
from navigation.srv import DirectionMove
from detection.srv import GetImage
from detection.srv import ObjectExist, ObjectExistResponse


def set_path():
    DEFAULT_PATH = "/home/mustar/test_ws/src/detection/GroundingDINO/"
    os.chdir(DEFAULT_PATH)

def calculate_centroid(x1, y1, x2, y2):
    centroid_x = (x1 + x2) / 2
    centroid_y = (y1 + y2) / 2
    print('centroids')
    print(centroid_x, centroid_y)
    return centroid_x, centroid_y

def plot_boxes_to_image(image_pil, tgt):
    H, W = tgt["size"]
    boxes = tgt["boxes"]
    labels = tgt["labels"]
    cent_y = cent_x = flag = 1 
    assert len(boxes) == len(labels), "boxes and labels must have same length"

    draw = ImageDraw.Draw(image_pil)
    mask = Image.new("L", image_pil.size, 0)
    mask_draw = ImageDraw.Draw(mask)

    # draw boxes and masks
    for box, label in zip(boxes, labels):
        # from 0..1 to 0..W, 0..H
        box = box * torch.Tensor([W, H, W, H])
        # from xywh to xyxy
        box[:2] -= box[2:] / 2
        box[2:] += box[:2]
        # random color
        color = tuple(np.random.randint(0, 255, size=3).tolist())
        # draw
        x0, y0, x1, y1 = box
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
        print(x0, y0, x1, y1)
        if flag:
            cent_x, cent_y = calculate_centroid(x0, y0, x1, y1)
            flag = 0

        draw.rectangle([x0, y0, x1, y1], outline=color, width=6)
        # draw.text((x0, y0), str(label), fill=color)

        font = ImageFont.load_default()
        if hasattr(font, "getbbox"):
            bbox = draw.textbbox((x0, y0), str(label), font)
        else:
            w, h = draw.textsize(str(label), font)
            bbox = (x0, y0, w + x0, y0 + h)
        # bbox = draw.textbbox((x0, y0), str(label))
        draw.rectangle(bbox, fill=color)
        draw.text((x0, y0), str(label), fill="white")

        mask_draw.rectangle([x0, y0, x1, y1], fill=255, width=6)

    return image_pil, mask, cent_x, cent_y


def load_image(image_path):
    # load image
    image_pil = Image.open(image_path).convert("RGB")  # load image

    # Display the image
    image_pil.show()

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image


def load_model(model_config_path, model_checkpoint_path, cpu_only=False):
    args = SLConfig.fromfile(model_config_path)
    args.device = "cuda" if not cpu_only else "cpu"
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model


def get_grounding_output(model, image, caption, box_threshold, text_threshold=None, with_logits=True, cpu_only=False, token_spans=None):
    rospy.loginfo('caption received {}'.format(caption))
    assert text_threshold is not None or token_spans is not None, "text_threshould and token_spans should not be None at the same time!"
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    device = "cuda" if not cpu_only else "cpu"
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"][0]  # (nq, 4)

    # filter output
    if token_spans is None:
        logits_filt = logits.cpu().clone()
        boxes_filt = boxes.cpu().clone()
        filt_mask = logits_filt.max(dim=1)[0] > box_threshold
        logits_filt = logits_filt[filt_mask]  # num_filt, 256
        boxes_filt = boxes_filt[filt_mask]  # num_filt, 4

        # get phrase
        tokenlizer = model.tokenizer
        tokenized = tokenlizer(caption)
        # build pred
        pred_phrases = []
        for logit, box in zip(logits_filt, boxes_filt):
            pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
            if with_logits:
                pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
            else:
                pred_phrases.append(pred_phrase)
    else:
        # given-phrase mode
        positive_maps = create_positive_map_from_span(
            model.tokenizer(text_prompt),
            token_span=token_spans
        ).to(image.device) # n_phrase, 256

        logits_for_phrases = positive_maps @ logits.T # n_phrase, nq
        all_logits = []
        all_phrases = []
        all_boxes = []
        for (token_span, logit_phr) in zip(token_spans, logits_for_phrases):
            # get phrase
            phrase = ' '.join([caption[_s:_e] for (_s, _e) in token_span])
            # get mask
            filt_mask = logit_phr > box_threshold
            # filt box
            all_boxes.append(boxes[filt_mask])
            # filt logits
            all_logits.append(logit_phr[filt_mask])
            if with_logits:
                logit_phr_num = logit_phr[filt_mask]
                all_phrases.extend([phrase + f"({str(logit.item())[:4]})" for logit in logit_phr_num])
            else:
                all_phrases.extend([phrase for _ in range(len(filt_mask))])
        boxes_filt = torch.cat(all_boxes, dim=0).cpu()
        pred_phrases = all_phrases


    return boxes_filt, pred_phrases


def pre_load():
    global model
    config_file = 'groundingdino/config/GroundingDINO_SwinT_OGC.py'  # change the path of the model config file
    checkpoint_path = 'weights/groundingdino_swint_ogc.pth'  # change the path of the model
    model = load_model(config_file, checkpoint_path, cpu_only=1)
    rospy.loginfo('model loaded')

def define_params(path, image_name):
    global model

     # cfg
    # /home/mustar/test_ws/src/detection/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py
    
    output_dir = 'Output_Pics'
    # image_path = '1.jpg'
    box_threshold = 0.3
    text_threshold = 0.25
    token_spans = None
    cpu_only = True
    image_path = os.path.join(path, image_name)


    # make dir
    os.makedirs(output_dir, exist_ok=True)
    # load image
    # print('hello')
    image_pil, image = load_image(image_path)
    rospy.loginfo('image loaded')
    # load model
    
    # rospy.loginfo('model loaded')

    # # visualize raw image
    # image_pil.save(os.path.join(output_dir, "raw_image.jpg"))

    # set the text_threshold to None if token_spans is set.
    if token_spans is not None:
        text_threshold = None
        print("Using token_spans. Set the text_threshold to None.")

    return image_pil, image, model, box_threshold, text_threshold, cpu_only, output_dir, token_spans

def turn_motor_service(movement):
    rospy.loginfo('waiting for motor service...')
    rospy.wait_for_service('move_motor')
    send = rospy.ServiceProxy('move_motor', DirectionMove)
    success = send(movement)
    print(success)
    return success

def centeralize_object(x, y): # move to center 250 -- 300 -- 350
    rospy.loginfo('centeralizing object...')
    movement = ''

    rospy.loginfo('value of x {}'.format(x))
    rospy.loginfo('value of y {}'.format(y))

    if x < 240: # turn left
        movement = 'left'
    elif x > 350: # turn right
        movement = 'right'
    else:
        return False
    rospy.loginfo('moving to {}'.format(movement))

    turn_motor_service(movement)
    return True


   

def main(image_pil, image, model, text_prompt, box_threshold, text_threshold, cpu_only, output_dir, token_spans, image_name, image_file_path):

    rospy.loginfo('get predicted output')
    # run model
    boxes_filt, pred_phrases = get_grounding_output(
        model, image, text_prompt, box_threshold, text_threshold, cpu_only=cpu_only, token_spans=eval(f"{token_spans}")
    )

    rospy.loginfo('visualizing prediction')

    # visualize pred
    size = image_pil.size
    pred_dict = {
        "boxes": boxes_filt,
        "size": [size[1], size[0]],  # H,W
        "labels": pred_phrases,
    }   

    # import ipdb; ipdb.set_trace()
    image_with_box, _, cent_x, cent_y = plot_boxes_to_image(image_pil, pred_dict)
    print("x, y {}{}".format(cent_x, cent_y))
    image_name = "{}_dino_pred.jpg".format(image_name[:-4])
    rospy.loginfo("image name saved {}".format(image_name))

    image_with_box.show()
    image_with_box.save(os.path.join(output_dir, 'pred.jpg'))
    
    rospy.loginfo('Detection finished')

    

    # Number of detected objects
    if len(boxes_filt): return True, cent_x, cent_y
    else: return False, None, None

def image_service(path):

    rospy.loginfo('waiting for get image service...')
    rospy.wait_for_service('imaging_server')

    send = rospy.ServiceProxy('imaging_server', GetImage)

    image_name = send(path)

    return image_name

def detect(request):

    image_file_path = '/home/mustar/test_ws/src/detection/GroundingDINO/Output_Pics'
    exist = off_center = 1
    while exist and off_center:
        exist = 0

        image_name = image_service(image_file_path)
        image_name = image_name.name

        rospy.loginfo('image name received {}'.format(image_name))

        image_pil, image, model, box_threshold, text_threshold, cpu_only, output_dir, token_spans = define_params(image_file_path, image_name)

        rospy.loginfo('Received request...')
        rospy.loginfo('object received {}'.format(request.object))
        exist, x, y = main(image_pil, image, model, request.object, box_threshold, text_threshold, cpu_only, output_dir, token_spans, image_name, image_file_path)
        rospy.loginfo('Request finished...')
        # x = y = 150
        if exist:
            rospy.loginfo('centering object...')
            off_center = 0
            # off_center = centeralize_object(x, y)
            print('off_center')
    # exist = 0
    return ObjectExistResponse(exist)



if __name__=="__main__":

    rospy.init_node('detection_server')
    rospy.loginfo('Detection Server Initiated')

    set_path()
    pre_load()

    try:
        service = rospy.Service('detect_object', ObjectExist, detect)
        
        rospy.spin()

    except rospy.ROSInterruptException:
        pass
