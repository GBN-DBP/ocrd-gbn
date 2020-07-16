import numpy as np
import cv2
import PIL.Image
import scipy.ndimage

def resolve_box(box):
    '''
    Retrieves the x0, y0, x1, y1 coordinates from the given box
    '''

    return box[0], box[1], box[0] + box[2], box[1] + box[3]

def box_to_polygon(box, offset=(0, 0)):
    '''
    Converts box to polygon (set of points)
    '''

    x0, y0, x1, y1 = resolve_box(box)

    # Apply offset:
    x0 += offset[0]
    y0 += offset[1]
    x1 += offset[0]
    y1 += offset[1]

    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]

def draw_box(image, box, color, thickness):
    '''
    Draws box rectangle on image
    '''

    x0, y0, x1, y1 = resolve_box(box)

    cv2.rectangle(image, (x0, y0), (x1, y1), color, thickness)

    return image

def draw_polygon(image, polygon, color, thickness):
    '''
    Draws polygon on image
    '''

    cv2.polylines(image, np.int32([polygon]), True, color, thickness)

    return image

def slice_image(image, box):
    '''
    Slices the given image in the x,y coordinates described in the given box
    '''

    x0, y0, x1, y1 = resolve_box(box)

    return image[y0:y1, x0:x1]

def invert_image(image):
    '''
    Makes white pixels black and black pixels white
    '''

    mask = (image / 255).astype(np.bool_)
    image = np.ones_like(image) * 255
    image[mask == True] = 0

    return image

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

    # Convert PIL image array to RGB then to Numpy array then to BGR (for OpenCV):
    image = cv2.cvtColor(np.array(image.convert('RGB'), dtype=np.uint8), cv2.COLOR_RGB2BGR)

    return image, alpha

def cv2_to_pil_rgb(image, alpha=None):
    '''
    Converts cv2 (OpenCV) BGR image to PIL RGB image
    '''
    # Convert OpenCV BGR image array (Numpy) to PIL RGB image with alpha channel:
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
    # Convert OpenCV grayscale image array (Numpy) to PIL grayscale image with alpha channel:
    image = PIL.Image.fromarray(image)

    # Restore alpha channel, if there is one:
    if alpha:
        image.putalpha(alpha)

    return image

def gray_to_bgr(image):
    '''
    Converts a grayscale cv2 image to BGR
    '''
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

def binary_to_mask(image):
    '''
    Converts a binary (grayscale) cv2 image to a Numpy mask
    '''
    # Map pixels from [0, 255] (grayscale) to [0, 1] (binary):
    image = image / 255.0

    # Convert image array to boolean (mask):
    image = image.astype(np.bool_)

    return image
