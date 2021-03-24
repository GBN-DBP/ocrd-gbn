import numpy as np
import cv2
import PIL.Image


def pil_to_cv2_rgb(image, bg_color=255):
    '''
    Converts PIL RGB image to cv2 (OpenCV) BGR image (Numpy array)
    '''
    # Remove alpha channel from image, if there is one:
    if image.mode == 'LA' or image.mode == 'RGBA':
        # Ensure RGBA:
        image = image.convert('RGBA')

        alpha = image.getchannel('A')

        # Paste image on a canvas:
        canvas = PIL.Image.new('RGBA', image.size, bg_color)
        canvas.paste(image, mask=alpha)

        image = canvas
    else:
        alpha = None

    # Convert PIL image array to RGB then to Numpy array then to BGR (for
    # OpenCV):
    image = cv2.cvtColor(
        np.array(
            image.convert('RGB'),
            dtype=np.uint8
        ),
        cv2.COLOR_RGB2BGR
    )

    return image, alpha


def cv2_to_pil_rgb(image, alpha=None):
    '''
    Converts cv2 (OpenCV) BGR image to PIL RGB image
    '''
    # Convert OpenCV BGR image array (Numpy) to PIL RGB image with alpha
    # channel:
    image = PIL.Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Restore alpha channel, if there is one:
    if alpha:
        image.putalpha(alpha)

    return image


def pil_to_cv2_gray(image, bg_color=255):
    '''
    Converts PIL grayscale image to cv2 (OpenCV) grayscale image (Numpy array)
    '''
    # Remove alpha channel from image, if there is one:
    if image.mode == 'LA' or image.mode == 'RGBA':
        # Ensure LA:
        image = image.convert('LA')

        alpha = image.getchannel('A')

        # Paste image on a canvas:
        canvas = PIL.Image.new('LA', image.size, bg_color)
        canvas.paste(image, mask=alpha)

        image = canvas
    else:
        alpha = None

    # Convert PIL image array to Numpy array (for OpenCV):
    image = np.array(image.convert('L'), dtype=np.uint8)

    return image, alpha


def cv2_to_pil_gray(image, alpha=None):
    '''
    Converts cv2 (OpenCV) grayscale image to PIL grayscale image
    '''
    # Convert OpenCV grayscale image array (Numpy) to PIL grayscale image with
    # alpha channel:
    image = PIL.Image.fromarray(image)

    # Restore alpha channel, if there is one:
    if alpha:
        image.putalpha(alpha)

    return image
