#NOTE: see this repo: https://github.com/EricPedley/FCRN-DepthPrediction for instructions on how to run
#example command: py predict.py NYU_FCRN-checkpoint/NYU_FCRN.ckpt image.jpg
import argparse
import os
import numpy as np
import tensorflow.compat.v1 as tf
from matplotlib import pyplot as plt
from PIL import Image

import depth_prediction.fcrn as models
tf.disable_v2_behavior()

class DepthMapGenerator:

    #no args
    def __init__(self):
        # Default input size
        height = 228
        width = 304
        #height and width for inputs into the network
        self.inputHeight=height
        self.inputWidth=width
        channels = 3
        batch_size = 1
        # Create a placeholder for the input image
        self.input_node = tf.placeholder(tf.float32, shape=(None, height, width, channels))
        self.net = models.ResNet50UpProj({'data': self.input_node}, batch_size, 1, False)
        self.sess=tf.Session()#https://www.tensorflow.org/api_docs/python/tf/compat/v1/Session
        # Use to load from ckpt file
        saver = tf.train.Saver()     
        saver.restore(self.sess, "depth_prediction/NYU_FCRN-checkpoint/NYU_FCRN.ckpt")

        # Use to load from npy file
        #net.load(model_data_path, sess)
        
    def __del__(self):
        self.sess.close()

    #img should be PIL image object
    def getPrediction(self,img):
        # Evalute the network for the given image
        img = img.resize([self.inputWidth,self.inputHeight], Image.ANTIALIAS)
        img = np.array(img).astype('float32')
        img = np.expand_dims(np.asarray(img), axis = 0)
        pred = self.sess.run(self.net.get_output(), feed_dict={self.input_node: img})
        return pred[0,:,:,0]

def predict(model_data_path, image_path,sess=None,net=None,input_node=None):

    
    # Default input size
    height = 228
    width = 304
    channels = 3
    batch_size = 1
   
    # Read image
    img = Image.open(image_path)
    img = img.resize([width,height], Image.ANTIALIAS)
    img = np.array(img).astype('float32')
    img = np.expand_dims(np.asarray(img), axis = 0)
   

    # Construct the network
    if net==None:
        # Create a placeholder for the input image
        input_node = tf.placeholder(tf.float32, shape=(None, height, width, channels))
        net = models.ResNet50UpProj({'data': input_node}, batch_size, 1, False)
    if sess==None:
        with tf.Session() as sess:

            # Load the converted parameters
            print('Loading the model')

            # Use to load from ckpt file
            saver = tf.train.Saver()     
            saver.restore(sess, model_data_path)

            # Use to load from npy file
            #net.load(model_data_path, sess) 

            # Evalute the network for the given image
            pred = sess.run(net.get_output(), feed_dict={input_node: img})
            return pred[0,:,:,0]
    else:
        pred = sess.run(net.get_output(), feed_dict={input_node: img})
        return pred[0,:,:,0]
        
                
def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', help='Converted parameters for the model')
    parser.add_argument('image_paths', help='Directory of images to predict')
    args = parser.parse_args()

    # Predict the image
    pred = predict(args.model_path, args.image_paths)
    with open("pred.npy","wb") as file:
        np.save(file,pred)
    # Plot result
    fig = plt.figure()
    ii = plt.imshow(pred, interpolation='nearest')
    fig.colorbar(ii)
    plt.show()
    
    os._exit(0)

if __name__ == '__main__':
    main()

        



