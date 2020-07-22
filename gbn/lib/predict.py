import math
import numpy as np
import cv2
import tensorflow as tf
import keras.models

class Predicting():
    '''
    Methods for predicting characteristics of images given a deep learning model
    '''
    def __init__(self, model_path, shaping):
        self.model_path = model_path
        self.shaping = shaping

        # Get default tensorflow session:
        self.session = tf.get_default_session()

        # If none, initiate a new session:
        if self.session is None:
            self.init_session()

        # Load Keras model:
        self.model = keras.models.load_model(model_path, compile=False)

        # Get input and output shapes of model, replacing None by 1:
        self.input_shape = (1, self.model.input_shape[1], self.model.input_shape[2], self.model.input_shape[3])
        self.output_shape = (1, self.model.output_shape[1], self.model.output_shape[2], self.model.output_shape[3])

        # Set predict() method to selected algorithm:
        if shaping == "resize":
            self.predict = self.predict_resize
        elif shaping == "split":
            self.predict = self.predict_split
        else:
            raise ValueError("Invalid shaping algorithm: {}".format(shaping))

    def init_session(self):
        '''
        Initiates a session allowing GPU growth
        '''
        cfg = tf.ConfigProto()
        cfg.gpu_options.allow_growth = True

        self.session = tf.InteractiveSession()

    def perform_prediction(self, image):
        '''
        Performs the actual prediction given an image whose shape matches the model input
        '''
        # Reshape image to model input shape (tensor):
        image = image.reshape(self.input_shape)

        # Perform prediction:
        prediction = self.model.predict(image)

        # Reshape prediction to image-like representation:
        prediction = prediction.reshape(self.output_shape[1:])

        # Convert prediction from displaying likeliness of all labels to displaying the most likely label in interval [0,1] (binary only):
        prediction = np.argmax(prediction[:, :, :2], axis=2).astype(np.uint8)

        return prediction

    def predict_resize(self, image):
        '''
        Performs a prediction on the given image by resizing it to the model input shape
        '''
        # Get original image shape:
        image_shape = image.shape

        # Map pixels from [0, 255] (grayscale) to [0, 1] (binary):
        image = image / 255.0

        # Resize image to input shape:
        image = cv2.resize(image, (self.input_shape[2], self.input_shape[1]), interpolation=cv2.INTER_NEAREST)

        # Perform prediction:
        prediction = self.perform_prediction(image)

        # Resize prediction to original image shape:
        return cv2.resize(prediction, (image_shape[1], image_shape[0]), interpolation=cv2.INTER_NEAREST)

    def predict_split(self, image):
        '''
        Performs a prediction on the given image by splitting it into patches with the same shape as the model input shape
        '''
        # Get original image shape:
        image_shape = image.shape

        # Patch shape is equal to the height and width of model input:
        patch_shape = (self.input_shape[1], self.input_shape[2])

        # Get padding for splitting image into equal-sized patches:
        padding = (patch_shape[0] - (image_shape[0] % patch_shape[0]), patch_shape[1] - (image_shape[1] % patch_shape[1]))

        # Split padding equally around the image:
        padding_top = math.floor(padding[0] / 2)
        padding_bottom = math.ceil(padding[0]/ 2)
        padding_left = math.floor(padding[1] / 2)
        padding_right = math.ceil(padding[1] / 2)

        # Apply padding:
        image = cv2.copyMakeBorder(image, padding_top, padding_bottom, padding_left, padding_right, cv2.BORDER_CONSTANT, (255, 255, 255))

        # Map pixels from [0, 255] (grayscale) to [0, 1] (binary):
        image = image / 255.0

        # Get number of patches per dimension:
        nyf = int(image.shape[0] / patch_shape[0])
        nxf = int(image.shape[1] / patch_shape[1])

        # Blank (black) image to write the patch predictions to:
        canvas = np.zeros(image.shape[:-1], dtype=np.uint8)

        for i in range(nxf):
            for j in range(nyf):
                # Get vertical coordinates of current patch:
                yd = j * patch_shape[0]
                yu = yd + patch_shape[0]

                # Get horizontal coordinates of current patch:
                xd = i * patch_shape[1]
                xu = xd + patch_shape[1]

                # Get patch:
                patch = image[yd:yu, xd:xu]

                # Perform prediction:
                prediction = self.perform_prediction(patch)

                # Add patch to canvas:
                canvas[yd:yd+patch_shape[0], xd:xd+patch_shape[1]] = prediction

        # Remove padding from canvas of predictions:
        prediction = canvas[padding_top+1:padding_top+1+image_shape[0], padding_left+1:padding_left+1+image_shape[1]]

        return prediction
