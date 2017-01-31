import argparse
import base64
import json

import numpy as np
import socketio
import eventlet
import eventlet.wsgi
import time
from PIL import Image
from PIL import ImageOps
from flask import Flask, render_template
from io import BytesIO

from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array

from scipy.misc import imread, imresize, toimage


### Code added for preprocessing the images ###################################
def crop_and_resize(image, crop=(55, 120), shape=(100, 100, 3)):
    """
    Crop and Resize images to given shape.
    """
    height, width, channels = shape
    image_resized = np.empty([height, width, channels])

    crop_top = crop[0]
    crop_bottom = crop[1]
    cropped_image = image[crop_top:crop_bottom, :, :]
    image_resized = imresize(cropped_image, shape)

    return image_resized

def preprocess(image):
    image = crop_and_resize(image)
    image = (image / 255. - .5).astype(np.float32) #normalize
    return image
###############################################################################

sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None

@sio.on('telemetry')
def telemetry(sid, data):
    # The current steering angle of the car
    steering_angle = data["steering_angle"]
    # The current throttle of the car
    throttle = data["throttle"]
    # The current speed of the car
    speed = data["speed"]
    # The current image from the center camera of the car
    imgString = data["image"]
    image = Image.open(BytesIO(base64.b64decode(imgString)))
    image_array = np.asarray(image)
    transformed_image_array = image_array[None, :, :, :]
    # This model currently assumes that the features of the model are just the images. Feel free to change this.

    ####
    # Resize and pre-process the same as in the notebook that trains the model!
    ####
    preprocessed_images = np.empty([len(transformed_image_array), 100, 100, 3])
    for i, img in enumerate(transformed_image_array):
        preprocessed_images[i] = preprocess(img)
    ####


    steering_angle = float(model.predict(preprocessed_images, batch_size=1))
    # The driving model currently just outputs a constant throttle. Feel free to edit this.
    throttle = 0.5
    print(steering_angle, throttle)
    send_control(steering_angle, throttle)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit("steer", data={
    'steering_angle': steering_angle.__str__(),
    'throttle': throttle.__str__()
    }, skip_sid=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument('model', type=str,
    help='Path to model definition json. Model weights should be on the same path.')
    args = parser.parse_args()
    with open(args.model, 'r') as jfile:
        model = model_from_json(jfile.read())

    model.compile("adam", "mse")
    weights_file = args.model.replace('json', 'h5')
    model.load_weights(weights_file)

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
