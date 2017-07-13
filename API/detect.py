import os
import facenet
import tensorflow as tf
import numpy as np
import math
import pickle
import uuid
import time
from scipy import misc
from sklearn.svm import SVC
import detect_face as df
import base64
import requests
import urllib
import socket
import cv2


class Image(object):
    def __init__(self, image_url=None, image_file=None, image_base64=None):
        self.image_url = image_url
        self.image_file = image_file
        self.image_base64 = image_base64
        self.error_message = None

    def read_image(self):
        if (not self.image_url) and (not self.image_file) and (not self.image_base64):
            self.error_message = '400 MISSING_ARGUMENTS: image'
            raise ValueError
        if self.image_file:
            tmp = self.image_file
        elif self.image_base64:
            tmp = base64.b64decode(self.image_base64)
        else:
            socket.setdefaulttimeout(5)
            try:
                response = urllib.urlopen(self.image_url)
            except:
                raise  # TODO
            tmp = np.asarray(bytearray(response.read()), dtype="uint8")
        try:
            tmp = cv2.imdecode(tmp, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY)
        except IOError:
            self.error_message = '400 IMAGE_ERROR_UNSUPPORTED_FORMAT: image'
            raise ValueError
        if img.shape[0] < 48 or img.shape[0] > 4096:
            self.error_message = '400 INVALID_IMAGE_SIZE: ' + str(img.shape[0])
            raise ValueError
        if img.shape[1] < 48 or img.shape[1] > 4096:
            self.error_message = '400 INVALID_IMAGE_SIZE: ' + str(img.shape[1])
            raise ValueError
        return img


def detect(image):
    time_start = time.time()
    minsize = 20  # minimum size of face
    gpu_memory_fraction = 0.5
    image_size = 160
    margin = 32
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor
    print('Creating networks and loading parameters')
    # with tf.Graph().as_default():
    #     gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
    #     sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    #     with sess.as_default():
    #         pnet, rnet, onet = df.create_mtcnn(sess, None)
    try:
        img = image.read_image()
    except ValueError:
        return image.error_message
    face_cascade = cv2.CascadeClassifier(
        '/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')
    landmarks = face_cascade.detectMultiScale(
        img,
        scaleFactor=1.15,
        minNeighbors=5,
        minSize=(5, 5),
        flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )
    faces = []
    for (x, y, w, h) in landmarks:
        cv2.rectangle(img, (x, y), (x + w, y + w), (0, 255, 0), 2)
        faces.append({
            "top": y,
            "left": x,
            "width": w,
            "height": h
        })
    # img_size = np.asarray(img.shape)[0:2]
    # bounding_boxes, _ = df.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
    #
    # det = np.squeeze(bounding_boxes[0, 0:4])
    # bb = np.zeros(4, dtype=np.int32)
    # bb[0] = np.maximum(det[0] - margin / 2, 0)
    # bb[1] = np.maximum(det[1] - margin / 2, 0)
    # bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
    # bb[3] = np.minimum(det[3] + margin / 2, img_size[0])
    # cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
    # aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
    image_id = str(uuid.uuid4())
    # if os.path.exists('/align'):  # TODO: Maybe link to database
    #     output_filename = os.path.join('/align', image_id + '.png')
    # else:
    #     os.mkdir('/align')
    #     output_filename = os.path.join('/align', image_id + '.png')
    # misc.imsave(output_filename, aligned)
    request_id = str(int(time.time()))
    time_stop = time.time()
    time_used = time_stop - time_start
    result = {
        'request_id': request_id,
        'faces:': faces,
        'image_id': image_id,
        'time_used': time_used
    }
    return result
