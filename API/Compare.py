import os
import shutil
import math
import pickle
import uuid
import time
import base64
import requests
import json
import cherrypy
from io import BytesIO
from scipy import misc
from sklearn.svm import SVC

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

import facenet
import detect_face 

tmp_dir = "/home/wzg/tmp/dataset"
model_path = "/home/wzg/models/20170512-110547.pb"
pkl_path = "/home/wzg/models/my_classifier.pkl"


def align(image, image_size=160, margin=22, gpu_memory_fraction=1.0):
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, None)

    minsize = 20   # minimum size of face
    threshold = [0.6, 0.7, 0.7]   # three steps's threshold
    factor = 0.709  # scale factor

    bounding_boxes, _ = detect_face.detect_face(image, minsize, pnet, rnet, onet, threshold, factor)
    nrof_faces = bounding_boxes.shape[0]

    if nrof_faces > 0:
        det = bounding_boxes[:, 0:4]
        img_size = np.asarray(image.shape)[0:2]
        if nrof_faces > 1:
            bounding_box_size = (det[:, 2]-det[:, 0]) * (det[:, 3]-det[:, 1])
            img_center = img_size / 2
            offsets = np.vstack(
                [(det[:, 0]+det[:, 2])/2-img_center[1], (det[:, 1]+det[:, 3])/2-img_center[0]])
            offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
            index = np.argmax(
                bounding_box_size-offset_dist_squared*2.0) # some extra weight on the centering
            det = det[index, :]
        det = np.squeeze(det)
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0]-margin/2, 0)
        bb[1] = np.maximum(det[1]-margin/2, 0)
        bb[2] = np.minimum(det[2]+margin/2, img_size[1])
        bb[3] = np.minimum(det[3]+margin/2, img_size[0])
        cropped = image[bb[1]:bb[3], bb[0]:bb[2], :]
        scaled = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
        prewhitened = facenet.prewhiten(scaled)
        return prewhitened
        #misc.imsave(output_filename, scaled)
    else:
        errorMessage = 'IMAGE_ALIGH_ERROR'
        print(errorMessage)


def face_from_face_token(face_token):
    face = []

    if os.path.exists(os.path.join(tmp_dir, face_token)):
        state = 1
        face = misc.imread(os.path.join(tmp_dir, face_token))
    else:
        state = 2

    return state, face


def image_from_file(file):
    image = []

    try:
        image_raw = file.file.read(8192*8192)
        image = misc.imread(BytesIO(image_raw))
    except IOError:
        state = 2
    else:
        state = 1
    return state, image


def image_from_base64(base64_file):
    image = []

    try:
        base64_de_file = base64.b64decode(base64_file)
        image = misc.imread(BytesIO(base64_de_file))
    except Exception, e:
        state = 2
    else:
        state = 1
    return state, image


def image_from_url(url):
    image = []

    try:
        r = requests.get(url, timeout=1.0)
    except requests.exceptions.ConnectTimeout:
        state = 3
    except requests.exceptions.ConnectionError:
        state = 2
    else:
        state = 1
        image = misc.imread(BytesIO(r.content))

    return state, image


def image_predict(img_list1, img_list2):
    with tf.Session() as sess:
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        embedding_size = embeddings.get_shape()[1]

        print('Calculating features for images')
        emb_array1 = np.zeros((1, embedding_size))
        emb_array2 = np.zeros((1, embedding_size))

        feed_dict = {images_placeholder:img_list1, phase_train_placeholder:False}
        emb_array1[:, :] = sess.run(embeddings, feed_dict=feed_dict)

        feed_dict = {images_placeholder:img_list2, phase_train_placeholder:False}
        emb_array2[:, :] = sess.run(embeddings, feed_dict=feed_dict)

        with open(pkl_path, 'rb') as infile:
            (model, class_names) = pickle.load(infile)

        print('Loaded classifier model from file "%s"' % pkl_path)

        predictions1 = model.predict_proba(emb_array1)
        predictions2 = model.predict_proba(emb_array2)

    return predictions1, predictions2, class_names



def compare(**kargs):
    re_dict = {}

    if 'face_token1' in kargs.keys():
        state1, face1 = face_from_face_token(kargs['face_token1'])
        if state1 == 1:
            img_list1 = [face1]
        else:
            errormessage = 'INVALID_FACE_TOKEN:{}'.format(kargs['face_token1'])
            re_dict['error_message'] = errormessage
            return re_dict

    elif 'image_file1' in kargs.keys():
        state1, image1 = image_from_file(kargs['image_file1'])
        if state1 == 1:
            face1 = align(image1)
            img_list1 = [face1]
        else:
            errormessage = 'IMAGE_ERROR_UNSUPPORTED_FORMAT:{}'.format('image_file1')
            re_dict['error_message'] = errormessage
            return re_dict

    elif 'image_base64_1' in kargs.keys():
        state1, image1 = image_from_base64(kargs['image_base64_1'])
        if state1 == 1:
            face1 = align(image1)
            img_list1 = [face1]
        else:
            errormessage = 'IMAGE_ERROR_UNSUPPORTED_FORMAT:{}'.format(kargs['image_base64_1'])
            re_dict['error_message'] = errormessage
            return re_dict

    elif 'image_url1' in kargs.keys():
        state1, image1 = image_from_url(kargs['image_url1'])
        if state1 == 1:
            face1 = align(image1)
            img_list1 = [face1]
        elif state1 == 2:
            errormessage = 'INVALID_IMAGE_URL:{}'.format(kargs['image_url1'])
            re_dict['error_message'] = errormessage
            return re_dict
        else:
            errormessage = 'IMAGE_DOWNLOAD_TIMEOUT:{}'.format(kargs['image_url1'])
            re_dict['error_message'] = errormessage
            return re_dict

    else:
        errormessage = 'MISSING ARGUMENTS:face_token1 or image_file1 \
                        or image_base64_1 or image_url1'
        re_dict['error_message'] = errormessage
        return re_dict


    if 'face_token2' in kargs.keys():
        state2, face2 = face_from_face_token(kargs['face_token2'])
        if state2 == 1:
            img_list2 = [face2]
        else:
            errormessage = 'INVALID_FACE_TOKEN:{}'.format(kargs['face_token2'])
            re_dict['error_message'] = errormessage
            return re_dict

    elif 'image_file2' in kargs.keys():
        state2, image2 = image_from_file(kargs['image_file2'])
        if state2 == 1:
            face2 = align(image2)
            img_list2 = [face2]
        else:
            errormessage = 'IMAGE_ERROR_UNSUPPORTED_FORMAT:{}'.format('image_file2')
            re_dict['error_message'] = errormessage
            return re_dict

    elif 'image_base64_2' in kargs.keys():
        state2, image2 = image_from_base64(kargs['image_base64_2'])
        if state2 == 1:
            face2 = align(image2)
            img_list2 = [face2]
        else:
            errormessage = 'IMAGE_ERROR_UNSUPPORTED_FORMAT:{}'.format(kargs['image_base64_2'])
            re_dict['error_message'] = errormessage
            return re_dict

    elif 'image_url2' in kargs.keys():
        state2, image2 = image_from_url(kargs['image_url2'])
        if state2 == 1:
            face2 = align(image2)
            img_list2 = [face2]
        elif state2 == 2:
            errormessage = 'INVALID_IMAGE_URL:{}'.format(kargs['image_url2'])
            re_dict['error_message'] = errormessage
            return re_dict
        else:
            errormessage = 'IMAGE_DOWNLOAD_TIMEOUT:{}'.format(kargs['image_url2'])
            re_dict['error_message'] = errormessage
            return re_dict

    else:
        errormessage = 'MISSING ARGUMENTS:face_token2 or image_file2 \
                        or image_base64_2 or image_url2'
        re_dict['error_message'] = errormessage
        return re_dict

    predictions1, predictions2, class_names = image_predict(img_list1, img_list2)

    best_class_indices1 = np.argmax(predictions1, axis=1)
    best_class_indices2 = np.argmax(predictions2, axis=1)

    re_dict = {
        '1st person': class_names[best_class_indices1[0]],
        '2nd person': class_names[best_class_indices2[0]]
    }

    return re_dict
