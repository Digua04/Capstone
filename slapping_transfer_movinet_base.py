
"""<table class="tfo-notebook-buttons" align="left">
  <td>
    <a target="_blank" href="https://www.tensorflow.org/tutorials/video/transfer_learning_with_movinet"><img src="https://www.tensorflow.org/images/tf_logo_32px.png" />View on TensorFlow.org</a>
  </td>
  <td>
    <a target="_blank" href="https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/video/transfer_learning_with_movinet.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png" />Run in Google Colab</a>
  </td>
  <td>
    <a target="_blank" href="https://github.com/tensorflow/docs/blob/master/site/en/tutorials/video/transfer_learning_with_movinet.ipynb"><img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />View source on GitHub</a>
  </td>
  <td>
    <a href="https://storage.googleapis.com/tensorflow_docs/docs/site/en/tutorials/video/transfer_learning_with_movinet.ipynb"><img src="https://www.tensorflow.org/images/download_logo_32px.png" />Download notebook</a>
  </td>
</table>

# Transfer learning for video classification with MoViNet

MoViNets (Mobile Video Networks) provide a family of efficient video classification models, supporting inference on streaming video. In this tutorial, you will use a pre-trained MoViNet model to classify videos, specifically for an action recognition task, from the [UCF101 dataset](https://www.crcv.ucf.edu/data/UCF101.php). A pre-trained model is a saved network that was previously trained on a larger dataset. You can find more details about MoViNets in the [MoViNets: Mobile Video Networks for Efficient Video Recognition](https://arxiv.org/abs/2103.11511) paper by Kondratyuk, D. et al. (2021). In this tutorial, you will:

* Learn how to download a pre-trained MoViNet model
* Create a new model using a pre-trained model with a new classifier by freezing the convolutional base of the MoViNet model
* Replace the classifier head with the number of labels of a new dataset
* Perform transfer learning on the [UCF101 dataset](https://www.crcv.ucf.edu/data/UCF101.php)

The model downloaded in this tutorial is from [official/projects/movinet](https://github.com/tensorflow/models/tree/master/official/projects/movinet). This repository contains a collection of MoViNet models that TF Hub uses in the TensorFlow 2 SavedModel format.

This transfer learning tutorial is the third part in a series of TensorFlow video tutorials. Here are the other three tutorials:

- [Load video data](https://www.tensorflow.org/tutorials/load_data/video): This tutorial explains much of the code used in this document; in particular, how to preprocess and load data through the `FrameGenerator` class is explained in more detail.
- [Build a 3D CNN model for video classification](https://www.tensorflow.org/tutorials/video/video_classification). Note that this tutorial uses a (2+1)D CNN that decomposes the spatial and temporal aspects of 3D data; if you are using volumetric data such as an MRI scan, consider using a 3D CNN instead of a (2+1)D CNN.
- [MoViNet for streaming action recognition](https://www.tensorflow.org/hub/tutorials/movinet): Get familiar with the MoViNet models that are available on TF Hub.

## Setup

Begin by installing and importing some necessary libraries, including:
[remotezip](https://github.com/gtsystem/python-remotezip) to inspect the contents of a ZIP file, [tqdm](https://github.com/tqdm/tqdm) to use a progress bar, [OpenCV](https://opencv.org/) to process video files (ensure that `opencv-python` and `opencv-python-headless` are the same version), and TensorFlow models ([`tf-models-official`](https://github.com/tensorflow/models/tree/master/official)) to download the pre-trained MoViNet model. The TensorFlow models package are a collection of models that use TensorFlow’s high-level APIs.
"""

# Commented out IPython magic to ensure Python compatibility.
import tqdm
import random
import pathlib
import imageio
import itertools
import collections
import os

import cv2
import numpy as np
import remotezip as rz
import seaborn as sns
# %matplotlib inline
import matplotlib.pyplot as plt
from tensorflow_docs.vis import embed
import PIL
import matplotlib as mpl
import mediapy as media

import keras
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy

# Import the MoViNet model from TensorFlow Models (tf-models-official) for the MoViNet model
from official.projects.movinet.modeling import movinet
from official.projects.movinet.modeling import movinet_model
from official.projects.movinet.tools import export_saved_model

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline
import matplotlib
matplotlib.use('TkAgg')

"""## Load data

The hidden cell below defines helper functions to download a slice of data from the UCF-101 dataset, and load it into a `tf.data.Dataset`. The [Loading video data tutorial](https://www.tensorflow.org/tutorials/load_data/video) provides a detailed walkthrough of this code.

The `FrameGenerator` class at the end of the hidden block is the most important utility here. It creates an iterable object that can feed data into the TensorFlow data pipeline. Specifically, this class contains a Python generator that loads the video frames along with its encoded label. The generator (`__call__`) function yields the frame array produced by `frames_from_video_file` and a one-hot encoded vector of the label associated with the set of frames.


"""

#@title
def format_frames(frame, output_size):
  """
    Pad and resize an image from a video.

    Args:
      frame: Image that needs to resized and padded.
      output_size: Pixel size of the output frame image.

    Return:
      Formatted frame with padding of specified output size.
  """
  frame = tf.image.convert_image_dtype(frame, tf.float32)
  frame = tf.image.resize_with_pad(frame, *output_size)
  return frame

def frames_from_video_file(video_path, n_frames, output_size = (224,224), frame_step = 5):
  """
    Creates frames from each video file present for each category.

    Args:
      video_path: File path to the video.
      n_frames: Number of frames to be created per video file.
      output_size: Pixel size of the output frame image.

    Return:
      An NumPy array of frames in the shape of (n_frames, height, width, channels).
  """
  # Read each video frame by frame
  result = []
  src = cv2.VideoCapture(str(video_path))

  video_length = src.get(cv2.CAP_PROP_FRAME_COUNT)

  need_length = 1 + (n_frames - 1) * frame_step

  if need_length > video_length:
    start = 0
  else:
    max_start = video_length - need_length
    start = random.randint(0, max_start + 1)

  src.set(cv2.CAP_PROP_POS_FRAMES, start)
  # ret is a boolean indicating whether read was successful, frame is the image itself
  ret, frame = src.read()
  result.append(format_frames(frame, output_size))

  for _ in range(n_frames - 1):
    for _ in range(frame_step):
      ret, frame = src.read()
    if ret:
      frame = format_frames(frame, output_size)
      result.append(frame)
    else:
      result.append(np.zeros_like(result[0]))
  src.release()
  result = np.array(result)[..., [2, 1, 0]]

  return result

def to_gif(images):
  converted_images = np.clip(images * 255, 0, 255).astype(np.uint8)
  imageio.mimsave('./animation.gif', converted_images, fps=10)
  return embed.embed_file('./animation.gif')

class FrameGenerator:
  def __init__(self, path, n_frames, training = False):
    """ Returns a set of frames with their associated label.

      Args:
        path: Video file paths.
        n_frames: Number of frames.
        training: Boolean to determine if training dataset is being created.
    """
    self.path = path
    self.n_frames = n_frames
    self.training = training
    self.class_names = sorted(set(p.name for p in self.path.iterdir() if p.is_dir()))
    self.class_ids_for_name = dict((name, idx) for idx, name in enumerate(self.class_names))

  def get_files_and_class_names(self):
    video_paths = list(self.path.glob('*/*.mp4'))
    classes = [p.parent.name for p in video_paths]
    return video_paths, classes

  def __call__(self):
    video_paths, classes = self.get_files_and_class_names()

    pairs = list(zip(video_paths, classes))

    if self.training:
      random.shuffle(pairs)

    for path, name in pairs:
      video_frames = frames_from_video_file(path, self.n_frames)
      label = self.class_ids_for_name[name] # Encode labels
      yield video_frames, label

"""Create the training and test datasets:"""

download_dir = pathlib.Path('./VR2_slapping/')
# download_dir = pathlib.Path('/content/drive/MyDrive/cap/VR2_slapping')
subset_paths = {'train': download_dir / 'train', 'test': download_dir / 'test',
                'val': download_dir / 'validate'}
batch_size = 8
num_frames = 8

CLASSES = sorted(os.listdir(subset_paths['train']))

output_signature = (tf.TensorSpec(shape = (None, None, None, 3), dtype = tf.float32),
                    tf.TensorSpec(shape = (), dtype = tf.int16))

train_ds = tf.data.Dataset.from_generator(FrameGenerator(subset_paths['train'], num_frames, training = True),
                                          output_signature = output_signature)
train_ds = train_ds.batch(batch_size)

val_ds = tf.data.Dataset.from_generator(FrameGenerator(subset_paths['val'], num_frames),
                                          output_signature = output_signature)
val_ds = val_ds.batch(batch_size)

test_ds = tf.data.Dataset.from_generator(FrameGenerator(subset_paths['test'], num_frames),
                                         output_signature = output_signature)
test_ds = test_ds.batch(batch_size)

"""The labels generated here represent the encoding of the classes. For instance, 'ApplyEyeMakeup' is mapped to the integer Take a look at the labels of the training data to ensure that the dataset has been sufficiently shuffled."""
fg = FrameGenerator(subset_paths['train'], num_frames, training = True)
label_names = list(fg.class_ids_for_name.keys())
print(label_names)

for frames, labels in train_ds.take(2):
  print(labels) 

"""Take a look at the shape of the data."""

print(f"Shape: {frames.shape}")
print(f"Label: {labels.shape}")

"""## Load MoViNets"""

gru = layers.GRU(units=4, return_sequences=True, return_state=True)

inputs = tf.random.normal(shape=[1, 10, 8]) # (batch, sequence, channels)
print(inputs)

result, state = gru(inputs) # Run it all at once

# Ensure correct state maintenance in sequence models
first_half, state = gru(inputs[:, :5, :])
print("First half output shape:", first_half.shape)
print("State shape:", state.shape)

batch_size = tf.shape(inputs)[0]
state = tf.reshape(state, (batch_size, -1))

second_half, _ = gru(inputs[:, 5:, :], initial_state=[state])
print("Second half output shape:", second_half.shape)

print(np.allclose(result[:, :5, :], first_half))
print(np.allclose(result[:, 5:, :], second_half))

"""## Download a pre-trained MoViNet model

In this section, you will:

1. You can create a MoViNet model using the open source code provided in [`official/projects/movinet`](https://github.com/tensorflow/models/tree/master/official/projects/movinet) from TensorFlow models.
2. Load the pretrained weights.
3. Freeze the convolutional base, or all other layers except the final classifier head, to speed up fine-tuning.

To build the model, you can start with the `a0` configuration because it is the fastest to train when benchmarked against other models. Check out the [available MoViNet models on TensorFlow Model Garden](https://github.com/tensorflow/models/blob/master/official/projects/movinet/configs/movinet.py) to find what might work for your use case.
"""
'''
# model_id = 'a2'
# resolution = 224

# tf.keras.backend.clear_session()

# backbone = movinet.Movinet(model_id=model_id)
# backbone.trainable = False

# # Set num_classes=600 to load the pre-trained weights from the original model
# model = movinet_model.MovinetClassifier(backbone=backbone, num_classes=600)
# model.build([None, None, None, None, 3])

# # Load pre-trained weights
# !wget https://storage.googleapis.com/tf_model_garden/vision/movinet/movinet_a2_base.tar.gz -O movinet_a0_base.tar.gz -q
# !tar -xvf movinet_a0_base.tar.gz

# checkpoint_dir = f'movinet_{model_id}_base'
# checkpoint_path = tf.train.latest_checkpoint(checkpoint_dir)
# checkpoint = tf.train.Checkpoint(model=model)
# status = checkpoint.restore(checkpoint_path)
# status.assert_existing_objects_matched()
'''
"""### Construct the backbone"""

model_id = 'a2'
resolution = 224
# use_positional_encoding = model_id in {'a3', 'a4', 'a5'}

tf.keras.backend.clear_session()

""" backbone = movinet.Movinet(
    model_id=model_id,
    causal=True,
    conv_type='2plus1d',
    se_type='2plus3d',
    activation='hard_swish',
    gating_activation='hard_sigmoid',
    use_positional_encoding=use_positional_encoding,
    use_external_states=False,
) """
backbone = movinet.Movinet(model_id=model_id)
backbone.trainable = False

# Set num_classes=600 to load the pre-trained weights from the original model
model = movinet_model.MovinetClassifier(
    backbone=backbone, num_classes=600)
model.build([None, None, None, None, 3])

"""### Load the pretrained weights"""
'''
# Extract pretrained weights
# !wget https://storage.googleapis.com/tf_model_garden/vision/movinet/movinet_a0_stream.tar.gz -O movinet_a0_stream.tar.gz -q
# !tar -xvf movinet_a0_stream.tar.gz
# !wget https://storage.googleapis.com/tf_model_garden/vision/movinet/movinet_a2_stream.tar.gz -O movinet_a2_stream.tar.gz -q
# !tar -xvf movinet_a2_stream.tar.gz
# import urllib.request
# import tarfile

# # 下载文件
# url = 'https://storage.googleapis.com/tf_model_garden/vision/movinet/movinet_a2_stream.tar.gz'
# filename = 'movinet_a2_stream.tar.gz'
# urllib.request.urlretrieve(url, filename)

# # 解压文件
# with tarfile.open(filename, 'r:gz') as tar:
#     tar.extractall()

# # 可选：删除下载的压缩文件
# os.remove(filename)
'''

checkpoint_dir = 'movinet_a2_base'
checkpoint_path = tf.train.latest_checkpoint(checkpoint_dir)
checkpoint = tf.train.Checkpoint(model=model)
status = checkpoint.restore(checkpoint_path)
status.assert_existing_objects_matched()

"""To build a classifier, create a function that takes the backbone and the number of classes in a dataset. The `build_classifier` function will take the backbone and the number of classes in a dataset to build the classifier. In this case, the new classifier will take a `num_classes` outputs (10 classes for this subset of UCF101)."""

def build_classifier(batch_size, num_frames, resolution, backbone, num_classes):
  """Builds a classifier on top of a backbone model."""
  model = movinet_model.MovinetClassifier(
      backbone=backbone,
      num_classes=num_classes)
  model.build([batch_size, num_frames, resolution, resolution, 3])

  return model

model = build_classifier(batch_size, num_frames, resolution, backbone, 2)

"""For this tutorial, choose the `tf.keras.optimizers.Adam` optimizer and the `tf.keras.losses.SparseCategoricalCrossentropy` 
loss function. Use the metrics argument to the view the accuracy of the model performance at every step."""

num_epochs = 3

loss_obj = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001)

model.compile(loss=loss_obj, optimizer='adam', metrics=['accuracy'])

"""Create a callback for storing the checkpoints"""
"""Save weights to .h5 file original
import os
home_dir = os.path.expanduser("~")
checkpoint_path = os.path.join(home_dir, "trained_model/cp.ckpt.weights.h5")
checkpoint_dir = os.path.dirname(checkpoint_path)
os.makedirs(checkpoint_dir, exist_ok=True)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1) """


"""### Train the model. After two epochs, observe a low loss with high accuracy for both the training and validation sets."""

results = model.fit(train_ds,
                    validation_data=val_ds,
                    epochs=num_epochs,
                    validation_freq=1,
                    verbose=1)

# 从本地MP4视频读取视频流
video_path = 'test_VR.mp4'  # 替换为本地MP4视频的路径
video_capture = cv2.VideoCapture(video_path)

# 视频编写器初始化
output_video_path = 'output_video.mp4'  # 输出视频文件的路径
fps = int(video_capture.get(cv2.CAP_PROP_FPS))  # 获取视频的帧率
frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))  # 获取视频帧的宽度
frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 获取视频帧的高度
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 设置视频编码器
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# 缓冲区，用于存储视频片段
buffer_size = 50  # 假设每2秒的片段包含50帧
frame_buffer = []

# 读取视频流并处理
while video_capture.isOpened():
    ret, frame = video_capture.read()
    if not ret:
        break

    # 预处理帧以适应模型
    frame_resized = cv2.resize(frame, (224, 224))
    frame_normalized = frame_resized / 255.0

    # 添加帧到缓冲区
    frame_buffer.append(frame_normalized)
    
    # 保持缓冲区的大小
    if len(frame_buffer) > buffer_size:
        frame_buffer.pop(0)

    # 只在缓冲区满了之后进行推理
    if len(frame_buffer) == buffer_size:
        # input_clip = np.expand_dims(np.array(frame_buffer), axis=0)  # 添加批次维度
        input_clip = np.expand_dims(np.array(frame_buffer, dtype=np.float32), axis=0)
        # 进行模型推理
        # predictions = loaded_model(input_clip)['logits']
        # predictions = model.signatures['serving_default'](tf.constant(input_clip, dtype=tf.float32))['Identity:0']
        predictions = model.predict(input_clip)
        predicted_class = np.argmax(predictions)
        predicted_class_name = label_names[predicted_class]

        # 显示结果在视频帧上
        cv2.putText(frame, f'Action: {predicted_class_name}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 将帧写入输出视频文件
        out.write(frame)

# 释放资源
video_capture.release()
out.release()
# cv2.destroyAllWindows()

"""## Evaluate the model

The model achieved high accuracy on the training dataset. Next, use Keras `Model.evaluate` to evaluate it on the test set.
"""

model.evaluate(test_ds, return_dict=True)

"""To visualize model performance further, use a [confusion matrix](https://www.tensorflow.org/api_docs/python/tf/math/confusion_matrix). The confusion matrix allows you to assess the performance of the classification model beyond accuracy. To build the confusion matrix for this multi-class classification problem, get the actual values in the test set and the predicted values."""

def get_actual_predicted_labels(dataset):
  """
    Create a list of actual ground truth values and the predictions from the model.

    Args:
      dataset: An iterable data structure, such as a TensorFlow Dataset, with features and labels.

    Return:
      Ground truth and predicted values for a particular dataset.
  """
  actual = [labels for _, labels in dataset.unbatch()]
  predicted = model.predict(dataset)

  actual = tf.stack(actual, axis=0)
  predicted = tf.concat(predicted, axis=0)
  predicted = tf.argmax(predicted, axis=1)

  return actual, predicted

def plot_confusion_matrix(actual, predicted, labels, ds_type):
  cm = tf.math.confusion_matrix(actual, predicted)
  ax = sns.heatmap(cm, annot=True, fmt='g')
  sns.set(rc={'figure.figsize':(12, 12)})
  sns.set(font_scale=1.4)
  ax.set_title('Confusion matrix of action recognition for ' + ds_type)
  ax.set_xlabel('Predicted Action')
  ax.set_ylabel('Actual Action')
  plt.xticks(rotation=90)
  plt.yticks(rotation=0)
  ax.xaxis.set_ticklabels(labels)
  ax.yaxis.set_ticklabels(labels)
  plt.show()

fg = FrameGenerator(subset_paths['train'], num_frames, training = True)
label_names = list(fg.class_ids_for_name.keys())
print(label_names)

actual, predicted = get_actual_predicted_labels(test_ds)
plot_confusion_matrix(actual, predicted, label_names, 'test')

# """## Reconstruct the whole model with use_external_states=True"""
""" 
model_id = 'a2'
resolution = 224
use_positional_encoding = model_id in {'a3', 'a4', 'a5'}

# tf.keras.backend.clear_session()

backbone = movinet.Movinet(
    model_id=model_id,
    causal=True,
    conv_type='2plus1d',
    se_type='2plus3d',
    activation='hard_swish',
    gating_activation='hard_sigmoid',
    use_positional_encoding=use_positional_encoding,
    use_external_states=True,
)
# backbone.trainable = False

model = movinet_model.MovinetClassifier(
    backbone=backbone, num_classes=2, output_states=True)
model.build([None, None, None, None, 3])

# Create the example input here.
# Refer to the paper for recommended input shapes.
model.build([None, None, None, None, 3])

# Load weights from the checkpoint to the rebuilt model
checkpoint_dir = 'trained_model'
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
"""
"""
## Inference using external states

def get_top_k(probs, k=5, label_map=CLASSES):
  # Outputs the top k model labels and probabilities on the given video.
  top_predictions = tf.argsort(probs, axis=-1, direction='DESCENDING')[:k]
  top_labels = tf.gather(label_map, top_predictions, axis=-1)
  top_labels = [label.decode('utf8') for label in top_labels.numpy()]
  top_probs = tf.gather(probs, top_predictions, axis=-1).numpy()
  return tuple(zip(top_labels, top_probs))

# Create initial states for the stream model
init_states_fn = model.init_states
init_states = init_states_fn(tf.shape(tf.ones(shape=[1, 1, resolution, resolution, 3])))

all_logits = []

# To run on a video, pass in one frame at a time
states = init_states
for frames, label in test_ds.take(1):
  for clip in frames[0]:
    # Input shape: [1, 1, 248, 248, 3]
    clip = tf.expand_dims(tf.expand_dims(clip, axis=0), axis=0)
    logits, states = model.predict({**states, 'image': clip}, verbose=0)
    all_logits.append(logits)

logits = tf.concat(all_logits, 0)
probs = tf.nn.softmax(logits)

final_probs = probs[-1]
top_k = get_top_k(final_probs)
print()
for label, prob in top_k:
  print(label, prob)

frames, label = list(test_ds.take(1))[0]
to_gif(frames[0].numpy())

## Animate the predictions over time

#@title
# Get top_k labels and probabilities predicted using MoViNets streaming model
def get_top_k_streaming_labels(probs, k=2, label_map=CLASSES):
  Returns the top-k labels over an entire video sequence.

  Args:
    probs: probability tensor of shape (num_frames, num_classes) that represents
      the probability of each class on each frame.
    k: the number of top predictions to select.
    label_map: a list of labels to map logit indices to label strings.

  Returns:
    a tuple of the top-k probabilities, labels, and logit indices
 
  top_categories_last = tf.argsort(probs, -1, 'DESCENDING')[-1, :1]
  # Sort predictions to find top_k
  categories = tf.argsort(probs, -1, 'DESCENDING')[:, :k]
  categories = tf.reshape(categories, [-1])

  counts = sorted([
      (i.numpy(), tf.reduce_sum(tf.cast(categories == i, tf.int32)).numpy())
      for i in tf.unique(categories)[0]
  ], key=lambda x: x[1], reverse=True)

  top_probs_idx = tf.constant([i for i, _ in counts[:k]])
  top_probs_idx = tf.concat([top_categories_last, top_probs_idx], 0)
  # find unique indices of categories
  top_probs_idx = tf.unique(top_probs_idx)[0][:k+1]
  # top_k probabilities of the predictions
  top_probs = tf.gather(probs, top_probs_idx, axis=-1)
  top_probs = tf.transpose(top_probs, perm=(1, 0))
  # collect the labels of top_k predictions
  top_labels = tf.gather(label_map, top_probs_idx, axis=0)
  # decode the top_k labels
  top_labels = [label.decode('utf8') for label in top_labels.numpy()]

  return top_probs, top_labels, top_probs_idx
"""
'''
# Plot top_k predictions at a given time step
def plot_streaming_top_preds_at_step(
    top_probs,
    top_labels,
    step=None,
    image=None,
    legend_loc='lower left',
    duration_seconds=10,
    figure_height=500,
    playhead_scale=0.8,
    grid_alpha=0.3):
  """Generates a plot of the top video model predictions at a given time step.

  Args:
    top_probs: a tensor of shape (k, num_frames) representing the top-k
      probabilities over all frames.
    top_labels: a list of length k that represents the top-k label strings.
    step: the current time step in the range [0, num_frames].
    image: the image frame to display at the current time step.
    legend_loc: the placement location of the legend.
    duration_seconds: the total duration of the video.
    figure_height: the output figure height.
    playhead_scale: scale value for the playhead.
    grid_alpha: alpha value for the gridlines.

  Returns:
    A tuple of the output numpy image, figure, and axes.
  """
  # find number of top_k labels and frames in the video
  num_labels, num_frames = top_probs.shape
  if step is None:
    step = num_frames
  # Visualize frames and top_k probabilities of streaming video
  fig = plt.figure(figsize=(6.5, 7), dpi=300)
  gs = mpl.gridspec.GridSpec(8, 1)
  ax2 = plt.subplot(gs[:-3, :])
  ax = plt.subplot(gs[-3:, :])
  # display the frame
  if image is not None:
    ax2.imshow(image, interpolation='nearest')
    ax2.axis('off')
  # x-axis (frame number)
  preview_line_x = tf.linspace(0., duration_seconds, num_frames)
  # y-axis (top_k probabilities)
  preview_line_y = top_probs

  line_x = preview_line_x[:step+1]
  line_y = preview_line_y[:, :step+1]

  for i in range(num_labels):
    ax.plot(preview_line_x, preview_line_y[i], label=None, linewidth='1.5',
            linestyle=':', color='gray')
    ax.plot(line_x, line_y[i], label=top_labels[i], linewidth='2.0')


  ax.grid(which='major', linestyle=':', linewidth='1.0', alpha=grid_alpha)
  ax.grid(which='minor', linestyle=':', linewidth='0.5', alpha=grid_alpha)

  min_height = tf.reduce_min(top_probs) * playhead_scale
  max_height = tf.reduce_max(top_probs)
  ax.vlines(preview_line_x[step], min_height, max_height, colors='red')
  ax.scatter(preview_line_x[step], max_height, color='red')

  ax.legend(loc=legend_loc)

  plt.xlim(0, duration_seconds)
  plt.ylabel('Probability')
  plt.xlabel('Time (s)')
  plt.yscale('log')

  fig.tight_layout()
  fig.canvas.draw()

  data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
  data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
  plt.close()

  figure_width = int(figure_height * data.shape[1] / data.shape[0])
  image = PIL.Image.fromarray(data).resize([figure_width, figure_height])
  image = np.array(image)

  return image

# Plotting top_k predictions from MoViNets streaming model
def plot_streaming_top_preds(
    probs,
    video,
    top_k=2,
    video_fps=25.,
    figure_height=500,
    use_progbar=True):
  """Generates a video plot of the top video model predictions.

  Args:
    probs: probability tensor of shape (num_frames, num_classes) that represents
      the probability of each class on each frame.
    video: the video to display in the plot.
    top_k: the number of top predictions to select.
    video_fps: the input video fps.
    figure_fps: the output video fps.
    figure_height: the height of the output video.
    use_progbar: display a progress bar.

  Returns:
    A numpy array representing the output video.
  """
  # select number of frames per second
  video_fps = 8.
  # select height of the image
  figure_height = 500
  # number of time steps of the given video
  steps = video.shape[0]
  # estimate duration of the video (in seconds)
  duration = steps / video_fps
  # estimate top_k probabilities and corresponding labels
  top_probs, top_labels, _ = get_top_k_streaming_labels(probs, k=top_k)

  images = []
  step_generator = tqdm.trange(steps) if use_progbar else range(steps)
  for i in step_generator:
    image = plot_streaming_top_preds_at_step(
        top_probs=top_probs,
        top_labels=top_labels,
        step=i,
        image=video[i],
        duration_seconds=duration,
        figure_height=figure_height,
    )
    images.append(image)

  return np.array(images)

# Generate a plot and output to a video tensor
video = test_ds.take(1)
video = list(video)[0][0]
video = video[2]
plot_video = plot_streaming_top_preds(probs, video, video_fps=8.)

media.show_video(plot_video, fps=3)
'''
## Save weights to .h5 file
# model.save_weights('movinet_a2_base_saved/model_weights.h5')
# model.load_weights('movinet_a2_base_saved/model_weights.h5')
"""## Export to saved model"""
# print("Start to save model!")
""" saved_model_dir = 'model'
tflite_filename = 'model.tflite'
input_shape = [None, None, None, None, 3]

export_saved_model.export_saved_model(
    model=model,
    input_shape=input_shape,
    export_path=saved_model_dir,
    causal=False,
    bundle_input_init_states_fn=False
) """
'''
# 包装模型以便导出为 SavedModel
class MoViNetWrapper(tf.Module):
    def __init__(self, model):
        self.model = model

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, None, 224, 224, 3], dtype=tf.float32)
    ])
    def __call__(self, input_tensor):
        return self.model(input_tensor)

movinet_wrapper = MoViNetWrapper(model)

# 导出为 SavedModel
saved_model_dir = 'trained_model'
tflite_filename = 'model.tflite'
tf.saved_model.save(movinet_wrapper, saved_model_dir)
print("~~~~~~~~~~~~~~Saved model!")

def get_top_k(probs, k=2, label_map=CLASSES):
  # Outputs the top k model labels and probabilities on the given video.
  top_predictions = tf.argsort(probs, axis=-1, direction='DESCENDING')[:k]
  top_labels = tf.gather(label_map, top_predictions, axis=-1)
  top_labels = [label.decode('utf8') for label in top_labels.numpy()]
  top_probs = tf.gather(probs, top_predictions, axis=-1).numpy()
  return tuple(zip(top_labels, top_probs))

"""## Convert to TF Lite"""

converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
tflite_model = converter.convert()

with open(tflite_filename, 'wb') as f:
  f.write(tflite_model)

# Create the interpreter and signature runner
interpreter = tf.lite.Interpreter(model_path=tflite_filename)
runner = interpreter.get_signature_runner()

init_states = {
    name: tf.zeros(x['shape'], dtype=x['dtype'])
    for name, x in runner.get_input_details().items()
}
del init_states['image']

"""## Inference using external states on tflite model"""

# To run on a video, pass in one frame at a time
states = init_states
for frames, label in test_ds.take(1):
  for clip in frames[0]:
    # Input shape: [1, 1, 172, 172, 3]
    outputs = runner(**states, image=clip)
    logits = outputs.pop('logits')[0]
    states = outputs

probs = tf.nn.softmax(logits)
top_k = get_top_k(probs)
print()
for label, prob in top_k:
  print(label, prob)

frames, label = list(test_ds.take(1))[0]
to_gif(frames[0].numpy())
'''
# """## Next steps

# Now that you have some familiarity with the MoViNet model and how to leverage various TensorFlow APIs (for example, for transfer learning), try using the code in this tutorial with your own dataset. The data does not have to be limited to video data. Volumetric data, such as MRI scans, can also be used with 3D CNNs. The NUSDAT and IMH datasets mentioned in [Brain MRI-based 3D Convolutional Neural Networks for Classification of Schizophrenia and Controls](https://arxiv.org/pdf/2003.08818.pdf) could be two such sources for MRI data.

# In particular, using the `FrameGenerator` class used in this tutorial and the other video data and classification tutorials will help you load data into your models.

# To learn more about working with video data in TensorFlow, check out the following tutorials:

# * [Load video data](https://www.tensorflow.org/tutorials/load_data/video)
# * [Build a 3D CNN model for video classification](https://www.tensorflow.org/tutorials/video/video_classification)
# * [MoViNet for streaming action recognition](https://www.tensorflow.org/hub/tutorials/movinet)
# """