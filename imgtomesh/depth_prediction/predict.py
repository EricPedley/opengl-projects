#NOTE: see this repo: https://github.com/EricPedley/FCRN-DepthPrediction for instructions on how to run
#example command: py predict.py NYU_FCRN-checkpoint/NYU_FCRN.ckpt image.jpg
import argparse
import os
import numpy as np
import tensorflow.compat.v1 as tf
from matplotlib import pyplot as plt
from PIL import Image

import fcrn as models
tf.disable_v2_behavior()

def simplePredict(img,session,net,input_node):
    # Default input size
    height = 228
    width = 304
    channels = 3
    batch_size = 1
    img = img.resize([width,height], Image.ANTIALIAS)
    img = np.array(img).astype('float32')
    img = np.expand_dims(np.asarray(img), axis = 0)
    pred = session.run(net.get_output(), feed_dict={input_node: img})
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

        



