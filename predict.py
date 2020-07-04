#$ python predict.py /path/to/image saved_model
#$ python predict.py /path/to/image saved_model --top_k K
#$ python predict.py /path/to/image saved_model --category_names map.json
import sys
import argparse
import warnings
warnings.filterwarnings('ignore')
import os
import numpy as np
from PIL import Image
import logging
import random
import json
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
from tensorflow.keras.preprocessing.image import ImageDataGenerator
tfds.disable_progress_bar()
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

# prepare defaults if used
dir = './test_images'
filename = random.choice(os.listdir("./test_images"))
image_path_def = os.path.join(dir, filename)

#parse data 
parser = argparse.ArgumentParser(description = "Image Classifier")
parser.add_argument("image_path", help= "Image Path", default= image_path_def)
parser.add_argument("model_path", help= "Model Path", default= "./ClemIV.h5")
parser.add_argument("--top_k", type=int, help="The top k predictions", required = False, default = 5)
parser.add_argument("--category_names", help="Class map json file", required = False, default = "./label_map.json")
args = parser.parse_args()
image_path = args.image_path 
saved_keras_model_filepath = args.model_path

# load the model
reloaded_keras_model = tf.keras.models.load_model(saved_keras_model_filepath, custom_objects={'KerasLayer':hub.KerasLayer})

#functions

def process_image(img):
    img = np.squeeze(img)
    image = tf.image.resize(img, (224, 224))   
    image = (image/255)
    return image

def predict(image_path,model,top_k):  
    image = Image.open(image_path, mode='r')
    image = np.asarray(image)
    image = process_image(image)
    image = np.expand_dims(image, axis = 0)
    ps = model.predict(image)
    probs, classes= tf.math.top_k(ps, top_k)
    classes += 1
    return probs, classes

probs, classes = predict(args.image_path, reloaded_keras_model, args.top_k)

#collect data

flowers = []
with open(args.category_names, 'r') as f:
    class_names = json.load(f)
    
for number in probs:
    number = number.numpy()
    performance = list(number)

for label in classes:
    label = label.numpy()
    label = list(label)
for i in label:
    name = str(i)
    flowers.append(class_names[name])

result = []
counter = 0

for (a, b) in zip(flowers, performance):
    counter +=1 
    ab = (counter, a, b)
    result.append(ab)    

# yoyoyo
print(result)


#execute!
predict(args.image_path, reloaded_keras_model, args.top_k)
