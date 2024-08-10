import time
import cv2
import tensorflow as tf
import numpy as np

# Load SavedModel
loaded_model = tf.saved_model.load('movinet_a2_base')
# loaded_model = tf.saved_model.load('movinet_a2_base_human')
# loaded_model = tf.saved_model.load('movinet_a2_base_cartoon')
serving_fn = loaded_model.signatures['serving_default']

# Define the class labels
label_names = ['Non-slapping', 'Slapping']

# Use the correct key for accessing the output
output_key = list(serving_fn.structured_outputs.keys())[0] # 'classifier_head_1'

# Load video
video_path = 'test_VR.mp4'
video_capture = cv2.VideoCapture(video_path)

# Define the output video path and create a VideoWriter object
output_video_path = 'output_video_latest.mp4'
fps = int(video_capture.get(cv2.CAP_PROP_FPS))
frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# Buffer to store frames
buffer_size = 50
frame_buffer = []
frame_count = 0

# Process video frame by frame
while video_capture.isOpened():
    frame_count += 1
    ret, frame = video_capture.read()
    if not ret:
        break

    # preprocess the frame
    frame_resized = cv2.resize(frame, (224, 224))
    frame_normalized = frame_resized / 255.0

    # add the frame to the buffer
    frame_buffer.append(frame_normalized)
    
    # keep the buffer size fixed
    if len(frame_buffer) > buffer_size:
        frame_buffer.pop(0)

    # if the buffer is full, perform inference
    if len(frame_buffer) == buffer_size:
        input_clip = np.expand_dims(np.array(frame_buffer, dtype=np.float32), axis=0)
        # start inference
        start_time = time.time()
        predictions = serving_fn(tf.constant(input_clip, dtype=tf.float32))[output_key]
        print(f"No. {frame_count} action inference time: ", time.time() - start_time)
        predicted_class = np.argmax(predictions.numpy())
        predicted_class_name = label_names[predicted_class]

        # draw the predicted action on the frame
        cv2.putText(frame, f'No. {frame_count} Action: {predicted_class_name}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)

        # write the frame to the output video
        out.write(frame)

# Release the video capture and video write objects
print('Finished processing video.')
video_capture.release()
out.release()
