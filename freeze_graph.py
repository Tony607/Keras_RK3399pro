#!/usr/bin/env python
# coding: utf-8

# ## Save the Keras model as a single .h5 file.

# In[1]:


# Force use CPU only.
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3 as Net
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import (
    preprocess_input,
    decode_predictions,
)
import numpy as np

print("TensorFlow version: {}".format(tf.__version__))

# Optional image to test model prediction.
img_path = "./data/elephant.jpg"
model_path = "./model"

# Path to save the model h5 file.
model_fname = os.path.join(model_path, "model.h5")

os.makedirs(model_path, exist_ok=True)

img_height = 299

model = Net(weights="imagenet", input_shape=(img_height, img_height, 3))


# Load the image for prediction.
img = image.load_img(img_path, target_size=(img_height, img_height))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)
# decode the results into a list of tuples (class, description, probability)
# (one such list for each sample in the batch)
print("Predicted:", decode_predictions(preds, top=3)[0])
# Predicted: [(u'n02504013', u'Indian_elephant', 0.82658225), (u'n01871265', u'tusker', 0.1122357), (u'n02504458', u'African_elephant', 0.061040461)]

# Save the h5 file to path specified.
model.save(model_fname)


# ## Benchmark Keras prediction speed.

# In[2]:


import time

times = []
for i in range(20):
    start_time = time.time()
    preds = model.predict(x)
    delta = time.time() - start_time
    times.append(delta)
mean_delta = np.array(times).mean()
fps = 1 / mean_delta
print("average(sec):{},fps:{}".format(mean_delta, fps))

# Clear any previous session.
tf.keras.backend.clear_session()


# ## Freeze graph
# Generate `.pb` file.

# In[3]:


import tensorflow as tf
from tensorflow.python.framework import graph_io
from tensorflow.keras.models import load_model


# Clear any previous session.
tf.keras.backend.clear_session()

save_pb_dir = "./model"
model_fname = "./model/model.h5"


def freeze_graph(
    graph,
    session,
    output,
    save_pb_dir=".",
    save_pb_name="frozen_model.pb",
    save_pb_as_text=False,
):
    with graph.as_default():
        graphdef_inf = tf.graph_util.remove_training_nodes(graph.as_graph_def())
        graphdef_frozen = tf.graph_util.convert_variables_to_constants(
            session, graphdef_inf, output
        )
        graph_io.write_graph(
            graphdef_frozen, save_pb_dir, save_pb_name, as_text=save_pb_as_text
        )
        return graphdef_frozen


# This line must be executed before loading Keras model.
tf.keras.backend.set_learning_phase(0)

model = load_model(model_fname)

session = tf.keras.backend.get_session()

INPUT_NODE = [t.op.name for t in model.inputs]
OUTPUT_NODE = [t.op.name for t in model.outputs]
print("\nINPUT_NODE: {}\nOUTPUT_NODE: {}".format(INPUT_NODE, OUTPUT_NODE))
frozen_graph = freeze_graph(
    session.graph,
    session,
    [out.op.name for out in model.outputs],
    save_pb_dir=save_pb_dir,
)


# ## Convert `.pb` file to RKNN model
#
# Run `convert_rknn.py`
