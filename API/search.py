import cherrypy
import os
import os.path
import facenet
import tensorflow as tf
import numpy as np
import pickle
import uuid
import time
from scipy import misc
import align.detect_face as df
import base64
import requests
from io import BytesIO

pkl_path = "/home/cp612sh/wsy/facenet/models/lfw_classifier.pkl" # Where to load the pickle
model_path = "/home/cp612sh/wsy/facenet/models/20170512-110547/20170512-110547.pb" # Where to load the model
train_path = "/home/cp612sh/test/train/" # Folder for training photos
align_path = "/home/cp612sh/test/align/" # Folder to store aligned photos
dirpath = "/home/cp612sh/test/photos" # Folder to store photos


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
    error_message="None"
    faces="None"
    image_id="None"
    #faceset_path = "None"
    aligned_images = 0
    img = []
    confidence = "None"
    result={}

    if 'faceset_token' in kwargs.keys():
        if os.path.exists(kwargs['faceset_token']):
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
                    time_used = time.time() - start_time
                    result['error_message'] = "INVALID_FACE_TOKEN"
                    result['request_id'] = request_id
                    result['time_used'] = time_used
                    return result
                faces, confidence = search_face(aligned_images)
        else:
            time_used = time.time() - start_time
            result['error_message'] = "INVALID_FACESET_TOKEN"
            result['request_id'] = request_id
            result['time_used'] = time_used
            return result

    elif 'outer_id' in kwargs.keys():
        if os.path.exists(kwargs['outer_id']):
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
                    time_used = time.time() - start_time
                    result['error_message'] = "INVALID_FACE_TOKEN"
                    result['request_id'] = request_id
                    result['time_used'] = time_used
                    return result
                faces, confidence = search_face(aligned_images)
        else:
            time_used = time.time() - start_time
            result['error_message'] = "INVALID_OUTER_ID"
            result['request_id'] = request_id
            result['time_used'] = time_used
            return result

    else:
        time_used = time.time() - start_time
        result['error_message'] = "MISSING_ARGUMENTS"
        result['request_id'] = request_id
        result['time_used'] = time_used
        return result


    if 'image_file' in kwargs.keys():
        image_id = str(uuid.uuid4())
        save_image = os.path.join(img_root, image_id + '.jpg')
        myfile = kwargs['image_file']
        try: 
            with open(save_image, 'wb') as newfile:
                while True:
                    data = myfile.file.read(8192)
                    if not data:
                        break
                    newfile.write(data)
        except IOError:
            time_used = time.time() - start_time
            result['error_message'] = "IMAGE_ERROR_UNSUPPORTED_FORMAT"
            result['request_id'] = request_id
            result['time_used'] = time_used
            return result
        img = misc.imread(save_image)
        if img.size > (max_img_size, max_img_size):
            time_used = time.time() - start_time
            result['error_message'] = "INVALID_IMAGE_SIZE"
            result['request_id'] = request_id
            result['time_used'] = time_used
            return result
        else:
            aligned_images = detect_face(img)
            faces, confidence = search_face(aligned_images)

    elif 'image_base64' in kwargs.keys():
        image_id = str(uuid.uuid4())
        save_image = os.path.join(img_root, image_id + '.jpg')
        myfile = kwargs['image_base64']
        try:
            with open(save_image, 'wb') as newfile:
                while True:
                    data = myfile.file.read(8192)
                    if not data:
                        break
                    imgdata = base64.b64decode(data)
                    newfile.write(imgdata)
        except IOError:
            time_used = time.time() - start_time
            result['error_message'] = "IMAGE_ERROR_UNSUPPORTED_FORMAT"
            result['request_id'] = request_id
            result['time_used'] = time_used
            return result
        img = misc.imread(save_image)
        if img.size > (max_img_size, max_img_size):
            time_used = time.time() - start_time
            result['error_message'] = "INVALID_IMAGE_SIZE"
            result['request_id'] = request_id
            result['time_used'] = time_used
            return result
        else:
            aligned_images = detect_face(img)
            faces, confidence = search_face(aligned_images)
                    
    elif 'image_url' in kwargs.keys():
        image_id = str(uuid.uuid4())
        save_image = os.path.join(img_root, image_id + '.jpg')
        try:  
            r = requests.get(kwargs['image_url'], timeout=1.0)
        except requests.exceptions.ConnectTimeout:
            time_used = time.time() - start_time
            result['error_message'] = "IMAGE_DOWNLOAD_TIMEOUT"
            result['request_id'] = request_id
            result['time_used'] = time_used
            return result
        except requests.exceptions.ConnectionError:
            time_used = time.time() - start_time
            result['error_message'] = "INVALID_IMAGE_URL"
            result['request_id'] = request_id
            result['time_used'] = time_used
            return result
        img = misc.imread(BytesIO(r.content))
        misc.imsave(save_image, img)
        if img.size > (max_img_size, max_img_size):
            time_used = time.time() - start_time
            result['error_message'] = "INVALID_IMAGE_SIZE"
            result['request_id'] = request_id
            result['time_used'] = time_used
            return result
        else:
            aligned_images = detect_face(img)
            faces, confidence = search_face(aligned_images)
            # pic_file = cStringIO.StringIO(urllib2.urlopen(kwargs['image_url']).read())
            # img = misc.imread(pic_file)
            # misc.imsave(save_image, img)
        
        
    else:
        time_used = time.time() - start_time
        result['error_message'] = "MISSING_ARGUMENTS"
        result['request_id'] = request_id
        result['time_used'] = time_used
        return result
        
            
    time_used = time.time() - start_time
    result['request_id'] = request_id
    result['image_id'] = image_id
    result['faces'] = faces
    result['confidence'] = confidence
    result['time_used'] = time_used
            
    return result


config = {
    'global' : {
        'server.socket_host' : '127.0.0.1',
        'server.socket_port' : 8080,
        'server.thread_pool' : 8,
        'server.max_request_body_size' : 0,
        'server.socket_timeout' : 60
  }
}

@cherrypy.expose
class API(object):
    
    @cherrypy.expose
    @cherrypy.tools.json_out()
    def index(self):
        return {"key": "value"}

    @cherrypy.expose
    @cherrypy.tools.json_out()
    def search(self, **kwargs):
        result = search(**kwargs)
        return result

    

    
if __name__ == '__main__':
    cherrypy.quickstart(API(), '/', config)


    