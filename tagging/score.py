

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

#change to script directory
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import argparse
import sys
import time

def load_graph(model_file):
  graph = tf.Graph()
  graph_def = tf.GraphDef()

  with open(model_file, "rb") as f:
    graph_def.ParseFromString(f.read())
  with graph.as_default():
    tf.import_graph_def(graph_def)

  return graph

def read_tensor_from_image_file(file_name, input_height=299, input_width=299,
				input_mean=0, input_std=255):
  input_name = "file_reader"
  output_name = "normalized"
  file_reader = tf.read_file(file_name, input_name)
  if file_name.endswith(".png"):
    image_reader = tf.image.decode_png(file_reader, channels = 3,
                                       name='png_reader')
  elif file_name.endswith(".gif"):
    image_reader = tf.squeeze(tf.image.decode_gif(file_reader,
                                                  name='gif_reader'))
  elif file_name.endswith(".bmp"):
    image_reader = tf.image.decode_bmp(file_reader, name='bmp_reader')
  else:
    image_reader = tf.image.decode_jpeg(file_reader, channels = 3,
                                        name='jpeg_reader')
  float_caster = tf.cast(image_reader, tf.float32)
  dims_expander = tf.expand_dims(float_caster, 0);
  resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
  normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
  sess = tf.Session()
  result = sess.run(normalized)

  return result

def load_labels(label_file):
  label = []
  proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
  for l in proto_as_ascii_lines:
    label.append(l.rstrip())
  return label



def run():
    file_name = "tf_files/test/a1.jpg"
    model_file = "tf_files/retrained_graph.pb"
    label_file = "tf_files/retrained_labels.txt"
    input_height = 224
    input_width = 224
    input_mean = 128
    input_std = 128
    input_layer = "input"
    output_layer = "final_result"

    #load model
    graph = load_graph(model_file)
    input_name = "import/" + input_layer
    output_name = "import/" + output_layer
    input_operation = graph.get_operation_by_name(input_name);
    output_operation = graph.get_operation_by_name(output_name);

    #load all test images
    images = []
    folderPath = "tf_files\\test"
    files = os.listdir(folderPath)
    for file in files:
        images.append(read_tensor_from_image_file(folderPath + "\\" + file, input_height=input_height, input_width=input_width, input_mean=input_mean, input_std=input_std))
    
    #score each image
    results = []
    with tf.Session(graph=graph) as sess:
        for image in images:
            results.append(sess.run(output_operation.outputs[0], {input_operation.outputs[0]: image}))

    #display results
    plt.figure(figsize=(len(results),1))
    labels = load_labels(label_file)
    template = "{} (score={:0.5f})"
    index = 0;
    for result in results:
        result = np.squeeze(result)
        top_k = result.argsort()[-5:][::-1]
        print('')
        print(files[index])
        for i in top_k:
            print(template.format(labels[i], result[i]))
        
        plt.subplot(1,len(results)+1,index+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[index][0], cmap=plt.cm.binary)
        plt.xlabel(labels[top_k[0]])
        index = index + 1
    plt.show();



run()