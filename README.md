# Detecting Actions Associated with Safety Risks Using Computer Vision in VRChat
This project aims to use a pre-trained MoViNet (Mobile Video Networks) model to perform binary classification of actions carried out by two avatars in VRChat.

## Create A New Model
The [slapping_transfer_movinet_base.py](https://github.com/Digua04/Capstone/blob/main/slapping_transfer_movinet_base.py) script is transferring learning for slapping detection using MoViNet.
The pre-trained MoViNet model is a saved network that was previously trained on [Kinetics-600 dataset](https://paperswithcode.com/dataset/kinetics-600). More details about MoViNets can be found in the [MoViNets: Mobile Video Networks for Efficient Video Recognition](https://arxiv.org/abs/2103.11511) paper by Kondratyuk, D. et al. (2021).

### VRChat Dataset Options
Three datasets, differentiated by the characteristics of avatars in the videos, have been created. To select a dataset for training your model, uncomment the corresponding line between 257 and 259.
```python
URL = 'https://github.com/Digua04/Capstone/raw/main/VR2_slapping.zip' # avatar features not specified
# URL = 'https://github.com/Digua04/Capstone/raw/main/human.zip' # avatars featuring human
# URL = 'https://github.com/Digua04/Capstone/raw/main/cartoon.zip' # avatars featuring cartoon
```
### Freeze the Convolutional Base of the MoViNet model
MoViNet-A2 serves as the backbone for creating a new model with an updated classifier. The convolutional base of the MoViNet model is frozen, and the classifier head is replaced with a new one that matches the number of labels in our dataset.

### Perform Transfer Learning on the VRChat dataset you selected aforementioned
After training, the model is saved as `movinet_a2_base` by default.

## Low Latency Inference
The [test_a2_base.py](https://github.com/Digua04/Capstone/blob/main/test_a2_base.py) script uses the saved model to predict avatar actions in a video. The inference latency per frame on an NVIDIA RTX 4090 GPU is approximately 39 ms, demonstrating the model's capability for real-time inference.
To select a saved model trained on the VR2_slapping, human, or cartoon dataset, uncomment the corresponding line below.
```python
loaded_model = tf.saved_model.load('movinet_a2_base')
# loaded_model = tf.saved_model.load('movinet_a2_base_human')
# loaded_model = tf.saved_model.load('movinet_a2_base_cartoon')
```

## Acknowledgements
Some of the code in this project is based on the [Transfer learning for video classification with MoViNet](https://www.tensorflow.org/tutorials/video/transfer_learning_with_movinet). We appreciate the guidance provided in this tutorial.