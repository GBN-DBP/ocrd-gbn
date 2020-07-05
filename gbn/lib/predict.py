import math
import numpy as np
import cv2
import tensorflow as tf
import keras.models

class Predicting():
    '''
    Methods for predicting characteristics of images given a deep learning model
    '''
    def __init__(self, model_path, prediction_algorithm):
        # Get default tensorflow session:
        self.session = tf.get_default_session()

        # If none, initiate a new session:
        if self.session is None:
            self.init_session()

        # Load Keras model:
        self.model = keras.models.load_model(model_path, compile=False)

        # Set predict() method to selected algorithm:
        if prediction_algorithm == "whole_image":
            self.predict = self.predict_whole_image
        elif prediction_algorithm == "sbb_patches":
            self.predict = self.predict_sbb_patches
        elif prediction_algorithm == "gbn_patches":
            self.predict = self.predict_gbn_patches
        else:
            raise ValueError("Invalid prediction algorithm: {}".format(prediction_algorithm))

    def init_session(self):
        '''
        Initiates a session allowing GPU growth
        '''
        cfg = tf.ConfigProto()
        cfg.gpu_options.allow_growth = True

        self.session = tf.InteractiveSession()

    def predict_whole_image(self, image):
        '''
        Applies model to given image by resshaping the input image to the model dimensions (as implemented originally on sbb_textline_detector)
        '''
        img_height_model = self.model.layers[-1].output_shape[1]
        img_width_model = self.model.layers[-1].output_shape[2]

        img = cv2.resize(image / 255.0, (img_width_model, img_height_model), interpolation=cv2.INTER_NEAREST)

        # Encapsulate image array inside a single-element array:
        label_p_pred = self.model.predict(img.reshape(1, img.shape[0], img.shape[1], img.shape[2]))

        # Reshape prediction into a 2-dimensional array with a single color channel (binary):
        seg = np.argmax(label_p_pred, axis=3)[0]

        return cv2.resize(seg, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST).astype(np.uint8)

    def predict_sbb_patches(self, image):
        '''
        Applies model to given image by splitting the input image in patches of the same dimensions as the model (as implemented originally on sbb_textline_detector)
        '''
        img_height_model = self.model.layers[-1].output_shape[1]
        img_width_model = self.model.layers[-1].output_shape[2]

        margin = int(0.1 * img_width_model)

        width_mid = img_width_model - 2 * margin
        height_mid = img_height_model - 2 * margin

        img = image / 255.0

        img_h = img.shape[0]
        img_w = img.shape[1]

        prediction_true = np.zeros((img_h, img_w, 3))

        mask_true = np.zeros((img_h, img_w))

        nxf = math.ceil(img_w / float(width_mid))
        nyf = math.ceil(img_h / float(height_mid))

        for i in range(nxf):
            for j in range(nyf):
                index_x_d = i * width_mid
                index_x_u = index_x_d + img_width_model

                index_y_d = j * height_mid
                index_y_u = index_y_d + img_height_model

                if index_x_u > img_w:
                    index_x_u = img_w
                    index_x_d = img_w - img_width_model
                if index_y_u > img_h:
                    index_y_u = img_h
                    index_y_d = img_h - img_height_model

                img_patch = img[index_y_d:index_y_u, index_x_d:index_x_u, :]

                label_p_pred = self.model.predict(img_patch.reshape(1, img_patch.shape[0], img_patch.shape[1], img_patch.shape[2]))

                seg = np.argmax(label_p_pred, axis=3)[0]
                seg_color = np.repeat(seg[:, :, np.newaxis], 3, axis=2)

                if i==0 and j==0:
                    seg_color = seg_color[0:seg_color.shape[0] - margin, 0:seg_color.shape[1] - margin, :]
                    seg = seg[0:seg.shape[0] - margin, 0:seg.shape[1] - margin]

                    mask_true[index_y_d + 0:index_y_u - margin, index_x_d + 0:index_x_u - margin] = seg
                    prediction_true[index_y_d + 0:index_y_u - margin, index_x_d + 0:index_x_u - margin,
                    :] = seg_color
                        
                elif i==nxf-1 and j==nyf-1:
                    seg_color = seg_color[margin:seg_color.shape[0] - 0, margin:seg_color.shape[1] - 0, :]
                    seg = seg[margin:seg.shape[0] - 0, margin:seg.shape[1] - 0]

                    mask_true[index_y_d + margin:index_y_u - 0, index_x_d + margin:index_x_u - 0] = seg
                    prediction_true[index_y_d + margin:index_y_u - 0, index_x_d + margin:index_x_u - 0,
                    :] = seg_color
                        
                elif i==0 and j==nyf-1:
                    seg_color = seg_color[margin:seg_color.shape[0] - 0, 0:seg_color.shape[1] - margin, :]
                    seg = seg[margin:seg.shape[0] - 0, 0:seg.shape[1] - margin]

                    mask_true[index_y_d + margin:index_y_u - 0, index_x_d + 0:index_x_u - margin] = seg
                    prediction_true[index_y_d + margin:index_y_u - 0, index_x_d + 0:index_x_u - margin,
                    :] = seg_color
                        
                elif i==nxf-1 and j==0:
                    seg_color = seg_color[0:seg_color.shape[0] - margin, margin:seg_color.shape[1] - 0, :]
                    seg = seg[0:seg.shape[0] - margin, margin:seg.shape[1] - 0]

                    mask_true[index_y_d + 0:index_y_u - margin, index_x_d + margin:index_x_u - 0] = seg
                    prediction_true[index_y_d + 0:index_y_u - margin, index_x_d + margin:index_x_u - 0,
                    :] = seg_color
                        
                elif i==0 and j!=0 and j!=nyf-1:
                    seg_color = seg_color[margin:seg_color.shape[0] - margin, 0:seg_color.shape[1] - margin, :]
                    seg = seg[margin:seg.shape[0] - margin, 0:seg.shape[1] - margin]

                    mask_true[index_y_d + margin:index_y_u - margin, index_x_d + 0:index_x_u - margin] = seg
                    prediction_true[index_y_d + margin:index_y_u - margin, index_x_d + 0:index_x_u - margin,
                    :] = seg_color
                        
                elif i==nxf-1 and j!=0 and j!=nyf-1:
                    seg_color = seg_color[margin:seg_color.shape[0] - margin, margin:seg_color.shape[1] - 0, :]
                    seg = seg[margin:seg.shape[0] - margin, margin:seg.shape[1] - 0]

                    mask_true[index_y_d + margin:index_y_u - margin, index_x_d + margin:index_x_u - 0] = seg
                    prediction_true[index_y_d + margin:index_y_u - margin, index_x_d + margin:index_x_u - 0,
                    :] = seg_color
                        
                elif i!=0 and i!=nxf-1 and j==0:
                    seg_color = seg_color[0:seg_color.shape[0] - margin, margin:seg_color.shape[1] - margin, :]
                    seg = seg[0:seg.shape[0] - margin, margin:seg.shape[1] - margin]

                    mask_true[index_y_d + 0:index_y_u - margin, index_x_d + margin:index_x_u - margin] = seg
                    prediction_true[index_y_d + 0:index_y_u - margin, index_x_d + margin:index_x_u - margin,
                    :] = seg_color
                        
                elif i!=0 and i!=nxf-1 and j==nyf-1:
                    seg_color = seg_color[margin:seg_color.shape[0] - 0, margin:seg_color.shape[1] - margin, :]
                    seg = seg[margin:seg.shape[0] - 0, margin:seg.shape[1] - margin]

                    mask_true[index_y_d + margin:index_y_u - 0, index_x_d + margin:index_x_u - margin] = seg
                    prediction_true[index_y_d + margin:index_y_u - 0, index_x_d + margin:index_x_u - margin,
                    :] = seg_color

                else:
                    seg_color = seg_color[margin:seg_color.shape[0] - margin, margin:seg_color.shape[1] - margin, :]
                    seg = seg[margin:seg.shape[0] - margin, margin:seg.shape[1] - margin]

                    mask_true[index_y_d + margin:index_y_u - margin, index_x_d + margin:index_x_u - margin] = seg
                    prediction_true[index_y_d + margin:index_y_u - margin, index_x_d + margin:index_x_u - margin,
                    :] = seg_color

        return prediction_true.astype(np.uint8)

    def predict_gbn_patches(self, image):
        '''
        Applies model to given image by splitting the input image in patches of the same dimensions as the model (without the bordering and redundancy from sbb_textline_detector)
        '''
        # Get model input dimensions:
        model_h = self.model.layers[-1].output_shape[1]
        model_w = self.model.layers[-1].output_shape[2]

        # Get patch dimensions (model input):
        patch_h = model_h
        patch_w = model_w

        # Map pixels from [0,1] (binary) to [0,255] (grayscale):
        image = image / 255.0

        # Get original image dimensions:
        img_h = image.shape[0]
        img_w = image.shape[1]

        # Get padding for splitting image into equal-sized patches:
        padding_h = patch_h - (img_h % patch_h)
        padding_w = patch_w - (img_w % patch_w)

        # Split padding equally around the image:
        padding_top = math.floor(padding_h / 2)
        padding_bottom = math.ceil(padding_h/ 2)
        padding_left = math.floor(padding_w / 2)
        padding_right = math.ceil(padding_w/ 2)

        # Apply padding:
        padded_img = cv2.copyMakeBorder(image, padding_top, padding_bottom, padding_left, padding_right, cv2.BORDER_CONSTANT, (255, 255, 255))

        # Get number of patches per dimension:
        nyf = int(padded_img.shape[0] / patch_h)
        nxf = int(padded_img.shape[1] / patch_w)

        # Predicted unbordered patches must be written to the canvas:
        canvas = np.zeros((padded_img.shape[0], padded_img.shape[1]))

        for i in range(nxf):
            for j in range(nyf):
                xd = i * patch_w
                xu = xd + patch_w

                yd = j * patch_h
                yu = yd + patch_h

                # Get patch:
                patch = padded_img[yd:yu, xd:xu]

                # Make prediction:
                pred = self.model.predict(patch.reshape(1, patch.shape[0], patch.shape[1], patch.shape[2]))

                # Reshape prediction into a 2-dimensional array with a single color channel (binary):
                pred = np.argmax(pred, axis=3)[0]

                # Add patch to canvas:
                canvas[yd:yd+patch_h, xd:xd+patch_w] = pred

        # Remove padding and return prediction image:
        return canvas[padding_top+1:padding_top+1+img_h, padding_left+1:padding_left+1+img_w].astype(np.uint8)
