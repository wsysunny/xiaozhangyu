import os
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

pkl_path = "/home/cp612sh/github/facenet/models/lfw_classifier.pkl" # Where to load the pickle
model_path = "/home/cp612sh/github/facenet/20170512-110547/20170512-110547.pb" # Where to load the model
train_path = "/home/cp612sh/test/train" # Folder for training photos
align_path = "/home/cp612sh/test/align" # Folder to store aligned photos
dirpath = "/home/cp612sh/test/photos" # Folder to store photos
batch_size = 100
image_size = 160
class GetPicture(pyinotify.ProcessEvent):
    def process_IN_CREATE(self, event): # Program will die here because we align the picture again and again and ...
        paths = [event.pathname]
        aligned_images = align(paths, image_size, 32, 0.5) # Align the picture
        recognize = Compare(paths, aligned_images)
        move_photos(recognize, align_path)
        train(model_path, pkl_path)

def align(image_paths, image_size, margin, gpu_memory_fraction):
    minsize = 20 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709 # scale factor
    
    print('Creating networks and loading parameters')
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = df.create_mtcnn(sess, None)
  
    nrof_samples = len(image_paths)
    img_list = [None] * nrof_samples
    for i in xrange(nrof_samples):
        img = misc.imread(os.path.expanduser(image_paths[i]))
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
        output_filename = os.path.join(align_path, time.asctime() + '.png')
        misc.imsave(output_filename, aligned)
        prewhitened = facenet.prewhiten(aligned)
        img_list[i] = prewhitened
    images = np.stack(img_list)
    return images

def Compare(paths, aligned_images):
    with tf.Graph().as_default():
        with tf.Session() as sess:
            facenet.load_model(model_path)
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]

            print('Calculating features for images')
            emb_array = np.zeros((1, embedding_size))
            paths_batch = paths[0:1]
            feed_dict = { images_placeholder:aligned_images, phase_train_placeholder:False }
            emb_array[0:1,:] = sess.run(embeddings, feed_dict=feed_dict)
            
            with open(pkl_path, 'rb') as infile:
                        (model, class_names) = pickle.load(infile)

            print('Loaded classifier model from file "%s"' % pkl_path)

            predictions = model.predict_proba(emb_array)
            print (predictions)
            compare_threshold = 0.5
            if max(predictions[0]) < compare_threshold:
                print('\nCan\'t recognize this person\n')
                return 0
            else:
                best_class_indices = np.argmax(predictions, axis=1)
                print('\nThe most likely person is: %s\n' % class_names[best_class_indices[0]])
                best_class_tmp = class_names[best_class_indices[0]]
                best_class = best_class_tmp.replace(' ', '_')
                return best_class

def move_photos(recognize, align_path):
    new_person = str(uuid.uuid4())
    filename = os.path.join(align_path, os.listdir(align_path)[0])
    if recognize == 0:
        while os.path.exists(os.path.join(train_path, new_person)):
            new_person = str(uuid.uuid4())
        new_folder = os.path.join(train_path, new_person)
        os.mkdir(new_folder)
        newname = os.path.join(new_folder, time.asctime() + '.png')
        shutil.move(filename, newname)
        for i in range(20):
            shutil.copy(newname, newname[:-4] + str(i) + '.png')
    else:
        shutil.move(filename, os.path.join(os.path.join(train_path, recognize), time.asctime() + '.png'))

            # best_class_indices = np.argmax(predictions, axis=1)
            # best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
            
            # for i in range(len(best_class_indices)):
            #     print('%4d  %s: %.3f' % (i, class_names[best_class_indices[i]], best_class_probabilities[i]))
                
            # accuracy = np.mean(np.equal(best_class_indices, labels))
            # print accuracy

def train(model_path, pkl_path):
    with tf.Graph().as_default():
        with tf.Session() as sess:
            dataset = facenet.get_dataset(train_path)
            for cls in dataset:
                assert(len(cls.image_paths)>0, 'There must be at least one image for each class in the dataset')
            
            paths, labels = facenet.get_image_paths_and_labels(dataset)
            
            print('Number of classes: %d' % len(dataset))
            print('Number of images: %d' % len(paths))
            # Load the model
            print('Loading feature extraction model')
            facenet.load_model(model_path)

            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]

            # Run forward pass to calculate embeddings
            print('Calculating features for images')
            nrof_images = len(paths)
            nrof_batches_per_epoch = int(math.ceil(1.0*nrof_images / batch_size))
            emb_array = np.zeros((nrof_images, embedding_size))
            for i in range(nrof_batches_per_epoch):
                start_index = i*batch_size
                end_index = min((i+1)*batch_size, nrof_images)
                paths_batch = paths[start_index:end_index]
                images = facenet.load_data(paths_batch, False, False, image_size)
                feed_dict = { images_placeholder:images, phase_train_placeholder:False }
                emb_array[start_index:end_index,:] = sess.run(embeddings, feed_dict=feed_dict)
            
            classifier_filename_exp = os.path.expanduser(pkl_path)
            print('Training classifier')
            model = SVC(kernel='linear', probability=True)
            model.fit(emb_array, labels)
            
            # Create a list of class names
            class_names = [ cls.name.replace('_', ' ') for cls in dataset]

            # Saving classifier model
            with open(classifier_filename_exp, 'wb') as outfile:
                pickle.dump((model, class_names), outfile)
            print('Saved classifier model to file "%s"' % classifier_filename_exp)
            
def watch():
    wm = pyinotify.WatchManager()
    wm.add_watch(dirpath, pyinotify.IN_CREATE, rec=True)
    gp = GetPicture()

    notifier = pyinotify.Notifier(wm, gp)
    notifier.loop()

def main():
    try:
        train(model_path, pkl_path)
    except:
        pass
    # with open(pkl_path, 'rb') as infile:
    #     (model, class_names) = pickle.load(infile)
    # print('Loaded classifier model from file "%s"' % pkl_path)
    watch()

if __name__ == '__main__':
	main()


    