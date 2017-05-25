import os
import pyinotify
import facenet
import tensorflow as tf
import numpy as np
import compare as align
import math
import pickle

pkl_path = "/home/cp612sh/models/my_classifier.pkl" # Where to load the pickle
model_path = "/home/mc/models/model-20170216-091149.pb" # Where to load the model
batch_size = 1000
image_size = 160
nrof_images = 1
labels = ["unknown"]
dirpath = "/home/mc/photos" # Folder to store picture
class GetPicture(pyinotify.ProcessEvent):
    def process_IN_CREATE(self, event): # Program will die here because we align the picture again and again and ...
        aligned_images = align.load_and_align_data([event.pathname], image_size, 32, 0.5) # Align the picture
        paths = [event.pathname]
        with tf.Graph().as_default():
            with tf.Session() as sess:
                facenet.load_model(model_path)
                images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
                embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
                phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
                embedding_size = embeddings.get_shape()[1]

                print('Calculating features for images')
                nrof_batches_per_epoch = int(math.ceil(1.0*nrof_images / batch_size))
                emb_array = np.zeros((nrof_images, embedding_size))
                for i in range(nrof_batches_per_epoch):
                    start_index = i*batch_size
                    end_index = min((i+1)*batch_size, nrof_images)
                    paths_batch = paths[start_index:end_index]
                    # images = facenet.load_data(paths_batch, False, False, image_size)
                    feed_dict = { images_placeholder:aligned_images, phase_train_placeholder:False }
                    emb_array[start_index:end_index,:] = sess.run(embeddings, feed_dict=feed_dict)
                
                with open(pkl_path, 'rb') as infile:
                            (model, class_names) = pickle.load(infile)

                print('Loaded classifier model from file "%s"' % pkl_path)

                predictions = model.predict_proba(emb_array)
                print (predictions)
                # if max < 0.6 we assume that the model can't recognize this people
                # take some photos and store them in train folders with a new label
                # retrain


                # best_class_indices = np.argmax(predictions, axis=1)
                # best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                
                # for i in range(len(best_class_indices)):
                #     print('%4d  %s: %.3f' % (i, class_names[best_class_indices[i]], best_class_probabilities[i]))
                    
                # accuracy = np.mean(np.equal(best_class_indices, labels))
                # print accuracy
        
def watch():
    wm = pyinotify.WatchManager()
    wm.add_watch(dirpath, pyinotify.IN_CREATE, rec=True)
    gp = GetPicture()

    notifier = pyinotify.Notifier(wm, gp)
    notifier.loop()

if __name__ == '__main__':
	watch()


    