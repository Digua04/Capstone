import tqdm
import random
import pathlib
import collections

import cv2
import numpy as np
import remotezip as rz
import seaborn as sns
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import keras
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers
import tarfile
from tensorflow.keras.optimizers import Adam

# Import the MoViNet model from TensorFlow Models (tf-models-official) for the MoViNet model
from official.projects.movinet.modeling import movinet
from official.projects.movinet.modeling import movinet_model


"""Load data
The hidden cell below defines helper functions to download a slice of data 
from the VR2_slapping dataset, and load it into a `tf.data.Dataset`.
The `FrameGenerator` class at the end of this block is the most important utility here. 
It creates an iterable object that can feed data into the TensorFlow data pipeline. 
Specifically, this class contains a Python generator that loads the video frames along with its encoded label. 
The generator (`__call__`) function yields the frame array produced by `frames_from_video_file` 
and a one-hot encoded vector of the label associated with the set of frames.
"""
def list_files_per_class(zip_url):
  """
    List the files in each class of the dataset given the zip URL.

    Args:
      zip_url: URL from which the files can be unzipped.

    Return:
      files: List of paths to files in the dataset.
  """
  files = []
  with rz.RemoteZip(zip_url) as zip:
    for zip_info in zip.infolist():
      if zip_info.filename[-1] == '/' or zip_info.filename.startswith('__MACOSX') \
        or zip_info.filename.startswith('.'):
        continue  # Skip directories and hidden files
      files.append(zip_info.filename)
  return files

def get_class(fname):
  """
    Retrieve the name of the class given a filename.

    Args:
      fname: Name of the file in the VR2_slapping dataset.

    Return:
      Class that the file belongs to.
  """
  return fname.split('_')[-3]

def get_files_per_class(files):
  """
    Retrieve the files that belong to each class.

    Args:
      files: List of files in the dataset.

    Return:
      Dictionary of class names (key) and files (values).
  """
  files_for_class = collections.defaultdict(list)
  for fname in files:
    class_name = get_class(fname)
    files_for_class[class_name].append(fname)
  for cls, file_list in files_for_class.items():
    print(f"Class {cls} has {len(file_list)} files")
  return files_for_class

def download_from_zip(zip_url, to_dir, file_names):
  """
    Download the contents of the zip file from the zip URL.

    Args:
      zip_url: Zip URL containing data.
      to_dir: Directory to download data to.
      file_names: Names of files to download.
  """
  with rz.RemoteZip(zip_url) as zip:
    for fn in tqdm.tqdm(file_names):
      class_name = get_class(fn)
      zip.extract(fn, str(to_dir / class_name))
      unzipped_file = to_dir / class_name / fn

      fn = pathlib.Path(fn).parts[-1]
      output_file = to_dir / class_name / fn
      unzipped_file.rename(output_file,)

def split_class_lists(files_for_class, count):
  """
    Returns the list of files belonging to a subset of data as well as the remainder of
    files that need to be downloaded.

    Args:
      files_for_class: Files belonging to a particular class of data.
      count: Number of files to download.

    Return:
      split_files: Files belonging to the subset of data.
      remainder: Dictionary of the remainder of files that need to be downloaded.
  """
  split_files = []
  remainder = {}
  print("Splitting files:", len(files_for_class))
  for cls in files_for_class:
    print(f"Class {cls} has {len(files_for_class[cls])} files")
    split_files.extend(files_for_class[cls][:count])
    remainder[cls] = files_for_class[cls][count:]
  return split_files, remainder

def download_vr2_slapping_subset(zip_url, num_classes, splits, download_dir):
  """
    Download a subset of the UFC101 dataset and split them into various parts, such as
    training, validation, and test.

    Args:
      zip_url: Zip URL containing data.
      num_classes: Number of labels.
      splits: Dictionary specifying the training, validation, test, etc. (key) division of data
              (value is number of files per split).
      download_dir: Directory to download data to.

    Return:
      dir: Posix path of the resulting directories containing the splits of data.
  """
  files = list_files_per_class(zip_url)
  files_for_class = get_files_per_class(files)

  classes = list(files_for_class.keys())[:num_classes]

  for cls in classes:
    new_files_for_class = files_for_class[cls]
    random.shuffle(new_files_for_class)
    files_for_class[cls] = new_files_for_class

  # Only use the number of classes you want in the dictionary
  files_for_class = {x: files_for_class[x] for x in list(files_for_class)[:num_classes]}

  dirs = {}
  for split_name, split_count in splits.items():
    print(split_name, ":")
    split_dir = download_dir / split_name
    split_files, files_for_class = split_class_lists(files_for_class, split_count)
    download_from_zip(zip_url, split_dir, split_files)
    dirs[split_name] = split_dir

  return dirs

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
    max_start = int(video_length - need_length)
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

"""Create the training, validation and testing datasets:"""
URL = 'https://github.com/Digua04/Capstone/raw/main/VR2_slapping.zip' # avatar features not specified
# URL = 'https://github.com/Digua04/Capstone/raw/main/human.zip' # avatars featuring human
# URL = 'https://github.com/Digua04/Capstone/raw/main/cartoon.zip' # avatars featuring cartoon
download_dir = pathlib.Path('./VR2_slapping/')
if URL == 'https://github.com/Digua04/Capstone/raw/main/VR2_slapping.zip':
  subset_paths = download_vr2_slapping_subset(URL,
                        num_classes = 2,
                        splits = {'train': 70, 'val': 15, 'test': 15},
                        download_dir = download_dir)
else:
  subset_paths = download_vr2_slapping_subset(URL,
                        num_classes = 2,
                        splits = {'train': 35, 'val': 8, 'test': 7},
                        download_dir = download_dir)

batch_size = 8
num_frames = 8

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

"""The labels generated here represent the encoding of the classes. 
For instance, 'Slapping' is mapped to the integer 1. 
Take a look at the labels of the training data to ensure that the dataset has been sufficiently shuffled."""
fg = FrameGenerator(subset_paths['train'], num_frames, training = True)
label_names = list(fg.class_ids_for_name.keys())
print(label_names)

for frames, labels in train_ds.take(3):
  print(labels) 

"""Take a look at the shape of the data."""

print(f"Shape: {frames.shape}")
print(f"Label: {labels.shape}")


"""## Test the GRU layer"""
gru = layers.GRU(units=4, return_sequences=True, return_state=True)
inputs = tf.random.normal(shape=[1, 10, 8]) # (batch, sequence, channels)

result, state = gru(inputs) # Run it all at once
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
1. Create a MoViNet model using the open source code provided in [`official/projects/movinet`]
(https://github.com/tensorflow/models/tree/master/official/projects/movinet) from TensorFlow models.
2. Load the pretrained weights.
3. Freeze the convolutional base, or all other layers except the final classifier head, to speed up fine-tuning.

To build the model, we can start with the `a2` configuration.
"""
"""Construct the backbone"""
model_id = 'a2'
resolution = 224

tf.keras.backend.clear_session()

backbone = movinet.Movinet(model_id=model_id)
backbone.trainable = False

# Set num_classes=600 to load the pre-trained weights from the original model
model = movinet_model.MovinetClassifier(backbone=backbone, num_classes=600)
model.build([None, None, None, None, 3])

# Load the pre-trained weights
import urllib.request

url = f'https://storage.googleapis.com/tf_model_garden/vision/movinet/movinet_a2_base.tar.gz'
filename = f'movinet_{model_id}_base.tar.gz'

urllib.request.urlretrieve(url, filename)

with tarfile.open(filename, 'r') as tar:
    tar.extractall()

checkpoint_dir = filename.split('.tar.gz')[0]
checkpoint_path = tf.train.latest_checkpoint(checkpoint_dir)
checkpoint = tf.train.Checkpoint(model=model)
status = checkpoint.restore(checkpoint_path)
status.assert_existing_objects_matched()

"""Build a classifier by creating a function that takes the backbone and the 
number of classes in the dataset.
"""
def build_classifier(batch_size, num_frames, resolution, backbone, num_classes):
  """Builds a classifier on top of a backbone model."""
  model = movinet_model.MovinetClassifier(
      backbone=backbone,
      num_classes=num_classes)
  model.build([batch_size, num_frames, resolution, resolution, 3])

  return model

model = build_classifier(batch_size, num_frames, resolution, backbone, 2)

"""Choose the tf.keras.optimizers.Adam optimizer and the tf.keras.losses.CategoricalCrossentropy loss function. 
Use the metrics argument to the view the accuracy of the model performance at every step."""
num_epochs = 2

loss_obj = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001)

model.compile(loss=loss_obj, optimizer='adam', metrics=['accuracy'])

# Train the model.
results = model.fit(train_ds,
                    validation_data=val_ds,
                    epochs=num_epochs,
                    validation_freq=1,
                    verbose=1)

# Evaluate the model on the test data.
print("Evaluating the model on the test data.")
results = model.evaluate(test_ds, return_dict=True)

# Save the model.
tf.saved_model.save(model, 'movinet_a2_base')
# tf.saved_model.save(model, 'movinet_a2_base_human_new')
# tf.saved_model.save(model, 'movinet_a2_base_cartoon_new')

"""To visualize model performance further, use a confusion matrix.
To build the confusion matrix for this multi-class classification problem, 
get the actual values in the test set and the predicted values."""
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
  sns.set(font_scale=1.5)
  ax.set_title('Confusion matrix of action recognition for ' + ds_type)
  ax.set_xlabel('Predicted Action')
  ax.set_ylabel('Actual Action')
  plt.xticks(rotation=90)
  plt.yticks(rotation=0)
  ax.xaxis.set_ticklabels(labels)
  ax.yaxis.set_ticklabels(labels)
  plt.show()

actual, predicted = get_actual_predicted_labels(test_ds)
plot_confusion_matrix(actual, predicted, label_names, 'test')
