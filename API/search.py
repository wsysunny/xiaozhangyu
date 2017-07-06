import random
import string

import cherrypy

import os
import os.path
import shutil
import pyinotify
import facenet
import tensorflow as tf
import numpy as np
import classifier as cf
import math
import pickle
import uuid
import time
from scipy import misc
from sklearn.svm import SVC
import align.detect_face as df
import urllib2
import Image
import cStringIO
import base64
import json
import requests
from io import BytesIO
import cv2

pkl_path = "/home/cp612sh/wsy/facenet/models/lfw_classifier.pkl" # Where to load the pickle
model_path = "/home/cp612sh/wsy/facenet/models/20170512-110547/20170512-110547.pb" # Where to load the model
train_path = "/home/cp612sh/test/train/" # Folder for training photos
align_path = "/home/cp612sh/test/align/" # Folder to store aligned photos
dirpath = "/home/cp612sh/test/photos" # Folder to store photos
error_dict = {'Error1':'parameter lost(image_url/image_file/image_base64 must have one)', 'Error2':'invalid face_token',
'Error3':'invalid faceset_token or outer_id','Error4':'image is oversize',
'Error5':'parameter lost(faceset_token/outer_id must have one)','Error6':'can not get image_file',
'Error7':'can not get image_base64','Error8':'can not align the image','Error9':'can not get image from web'}

img_root = '/home/cp612sh/'

max_img_size = 1500


    

def detect_face(img):
    minsize = 20 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709 # scale factor
    margin = 32
    image_size = 160


    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = df.create_mtcnn(sess, None)
    
    img_size = np.asarray(img.shape)[0:2]
    bounding_boxes, _ = df.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
    det = np.squeeze(bounding_boxes[0,0:4])
    bb = np.zeros(4, dtype=np.int32)
    bb[0] = np.maximum(det[0]-margin/2, 0)
    bb[1] = np.maximum(det[1]-margin/2, 0)
    bb[2] = np.minimum(det[2]+margin/2, img_size[1])
    bb[3] = np.minimum(det[3]+margin/2, img_size[0])
    cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
    aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
    face_token = str(uuid.uuid4())
    output_filename = os.path.join(align_path, face_token + '.png')
    misc.imsave(output_filename, aligned)
    prewhitened = facenet.prewhiten(aligned)
    aligned_image = prewhitened
    aligned_pictures = [aligned_image]
       
    return aligned_pictures

def search_face(aligned_images):
    with tf.Graph().as_default():
        with tf.Session() as sess:
            facenet.load_model(model_path)
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]

            emb_array = np.zeros((1, embedding_size))
                                
            feed_dict = { images_placeholder:aligned_images, phase_train_placeholder:False }
            emb_array[0:1,:] = sess.run(embeddings, feed_dict=feed_dict)
                                
            with open(pkl_path, 'rb') as infile:  # pkl_path is related to faceset_path
                (model, class_names) = pickle.load(infile)
            predictions = model.predict_proba(emb_array)
                                
            #best_class_indices = np.argmax(predictions, axis=1)
            best_class_indices = np.argsort(-predictions, axis=1)
            faces = []
            confidence = []
            for i in xrange(5):
                faces.append(class_names[best_class_indices[0][i]])
                confidence.append(predictions[0][best_class_indices[0][i]])
    return faces, confidence    


def search(**kwargs):

    start_time = time.time()
    request_id = str(uuid.uuid4())
    error_message = []
    faces="None"
    image_id="None"
    #faceset_path = "None"
    aligned_images = 0
    img = 0
    confidence = "None"

   
    if 'faceset_token' in kwargs.keys() or 'outer_id' in kwargs.keys(): 
        if 'faceset_token' in kwargs.keys():
            faceset_path = kwargs['faceset_token']
        elif 'outer_id' in kwargs.keys():
            faceset_path = kwargs['outer_id']

        if os.path.exists(faceset_path):
            if 'face_token' in kwargs.keys():
                mark = 0
                image_id = kwargs['face_token']
                for parent, dirnames, filenames in os.walk(faceset_path):
                    for filename in filenames:
                        if filename == kwargs['face_token'] :
                            aligned_image = misc.imread(os.path.join(parent, filename))
                            prewhitened = facenet.prewhiten(aligned_image)
                            aligned_images = [prewhitened]
                            mark = 1
                if mark == 0:
                    error_message.append(error_dict['Error2'])
                faces, confidence = search_face(aligned_images)
            else:
                if 'image_file' in kwargs.keys():
                    image_id = str(uuid.uuid4())
                    save_image = os.path.join(img_root, image_id + '.jpg')
                    myfile = kwargs['image_file']
                    with open(save_image, 'wb') as newfile:
                        while True:
                            data = myfile.file.read(8192)
                            if not data:
                                break
                            newfile.write(data)
                    
                    img = misc.imread(save_image)
                    #img = cv2.imread(save_image)
                   

                elif 'image_base64' in kwargs.keys():
                    image_id = str(uuid.uuid4())
                    save_image = os.path.join(img_root, image_id + '.jpg')
                    myfile = kwargs['image_base64']
                    with open(save_image, 'wb') as newfile:
                        while True:
                            data = myfile.file.read(8192)
                            if not data:
                                break
                            imgdata = base64.b64decode(data)
                            newfile.write(imgdata)
                    
                    img = misc.imread(save_image)
                    #img = cv2.imread(save_image)
                    # with open(os.path.join(img_root, kwargs['image_base64']), 'r') as fin:
                    #     img64data = fin.read()

                    #     imgdata = base64.b64decode(img64data) #image_base64 here is string
                    #     image_id = str(uuid.uuid4())
                    #     f = open(os.path.join(img_root, image_id + '.jpg'), 'wb') 
                    #     f.write(imgdata)
                    #     f.close()
                    #     img = misc.imread(os.path.join(img_root,image_id + '.jpg'))

                elif 'image_url' in kwargs.keys():
                    
                    # response = requests.get(kwargs['image_url'], timeout=10)
                    # pic_file = BytesIO(response.content)
                    
                    pic_file = cStringIO.StringIO(urllib2.urlopen(kwargs['image_url']).read())
                    imgdata = misc.imread(pic_file)
                    image_id = str(uuid.uuid4())
                    save_image = os.path.join(img_root, image_id + '.jpg')
                    misc.imsave(save_image, imgdata)
                    img = cv2.imread(save_image)
                else:
                    error_message.append(error_dict['Error1'])
                    

                if img.size > (max_img_size, max_img_size):
                    error_message.append(error_dict['Error4'])
                else:
                    aligned_images = detect_face(img)
                    faces, confidence = search_face(aligned_images)
            
            
        
        else:
            error_message.append(error_dict['Error3'])
    
    else:
        error_message.append(error_dict['Error5'])
            
    time_used = time.time() - start_time
    result = {"request_id":request_id,"image_id":image_id,"faces":faces,"confidence":confidence,"time_used":time_used, "error_message":error_message}
            
    return result




@cherrypy.expose
class StringGeneratorWebService(object):
    @cherrypy.expose

    @cherrypy.tools.accept(media='text/plain')
    def GET(self):
        return cherrypy.session['mystring']

    def POST(self, **kwargs):

        result = search(**kwargs)
        
        
       # cherrypy.session['mystring'] = {param1, param2}
        return json.dumps(result)

    def PUT(self, param1, param2):
        cherrypy.session['mystring'] = {param2,param1}


    def DELETE(self):
        cherrypy.session.pop('mystring', None)


if __name__ == '__main__':
    conf = {
        '/': {
            'request.dispatch': cherrypy.dispatch.MethodDispatcher(),
            'tools.sessions.on': True,
            'tools.response_headers.on': True,
            'tools.response_headers.headers': [('Content-Type', 'application/json')],
        }
    }
    cherrypy.quickstart(StringGeneratorWebService(), '/', conf)
    