# -*- coding: utf-8 -*-

from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image

import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import time
import json
from retina.utils import visualize_boxes

MODEL_PATH = 'snapshots/resnet50_full.h5'
IMAGE_PATH = 'samples/JPEGImages/2.png'

def load_inference_model(model_path=os.path.join('snapshots', 'resnet.h5')):
    model = models.load_model(model_path, backbone_name='resnet50')
    model = models.convert_model(model)
    model.summary()
    return model

def post_process(boxes, original_img, preprocessed_img):
    # post-processing
    h, w, _ = preprocessed_img.shape
    h2, w2, _ = original_img.shape
    boxes[:, :, 0] = boxes[:, :, 0] / w * w2
    boxes[:, :, 2] = boxes[:, :, 2] / w * w2
    boxes[:, :, 1] = boxes[:, :, 1] / h * h2
    boxes[:, :, 3] = boxes[:, :, 3] / h * h2
    return boxes


if __name__ == '__main__':
    
    model = load_inference_model(MODEL_PATH)
    dicts = []
    for i in range(1, 13069):
        data = {"bbox": [], 'label': [], 'score': []}
        # load image
        image = read_image_bgr('/home/haaaaa/tf2-eager-yolo3/tests/dataset/svhn/test/'+str(i)+'.png')

        # copy to draw on
        draw = image.copy()
        draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

        # preprocess image for network
        image = preprocess_image(image)
        image, _ = resize_image(image, 416, 448)

        # process image
        start = time.time()
        boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
        print("processing time: ", time.time() - start)

        boxes = post_process(boxes, draw, image)
        labels = labels[0]
        scores = scores[0]
        boxes = boxes[0]

        visualize_boxes(draw, boxes, labels, scores, data, class_labels=['10', '1', '2', '3', '4', '5', '6', '7', '8', '9'])

        print(data)
        dicts.append(data)
        
    dumped = json.dumps(dicts)
    with open('data.json', 'w') as fp:
        json.dump(dicts, fp)
    fp.close()
    '''
    # 5. plot    
    plt.imshow(draw)
    plt.show()
    '''