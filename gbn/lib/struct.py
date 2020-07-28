import numpy as np
import cv2
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

class bbox:
    '''
    Represents the bounding box of a shape in an image.
    '''

    def __init__(self, rectangle):
        '''
        Constructs a bbox object from a cv2 rectangle (x, y, w, h).
        '''

        # Parse the rectangle:
        self.x0, self.y0, self.width, self.height = rectangle

        # Calculate bounds on each axis:
        self.x1 = self.x0 + self.width
        self.y1 = self.y0 + self.height

        # Calculate area of bounding box:
        self.area = self.width * self.height

    @classmethod
    def from_points(self, points):
        '''
        Constructs a bbox object from a list of points through cv2.boundingRect.
        '''

        return bbox(cv2.boundingRect(points))

    def split(self, intervals, axis):
        '''
        Splits the bounding box along given axis given a list of intervals. An interval consist of a tuple (start, end)
        representing the interval [start, end[ along the axis to be split.
        '''

        boxes = []
        if axis:
            # On y-axis (1):
            for y0, y1 in intervals:
                boxes.append(bbox([self.x0, y0, self.width, y1 - y0]))
        else:
            # on x-axis (0):
            for x0, x1 in intervals:
                boxes.append(bbox([x0, self.y0, x1 - x0, self.height]))

        return boxes

class contour:
    '''
    Wrapper of cv2 contour (shape of an image).
    '''

    def __init__(self, points, hierarchy):
        '''
        Constructs a contour object. Both the points and the hierarchy returned from the cv2.findContours call with
        mode cv2.RETR_TREE must be provided.
        '''

        self.points = points

        # Parse hierarchy from cv2 array:
        self.next, self.previous, self.first_child, self.parent = hierarchy

        # Extract bounding box of contour:
        self.box = bbox.from_points(self.points)

        # Map points to origin (x0 == 0, y0 == 0):
        self.mapped_points = np.stack((self.points[:, 0] - self.box.x0, self.points[:, 1] - self.box.y0), axis=1)

        # Get area of contour:
        self.area = cv2.contourArea(self.points)

    def is_child(self):
        '''
        Checks whether the contour is a child of another contour along the hierarchy.
        '''

        return self.parent != -1

    def is_polygon(self):
        '''
        Checks whether the contour composes a valid polygon.
        '''

        return len(self.points) >= 3

    def to_mask(self):
        '''
        Converts contour to mask.
        '''

        # Create a background canvas with the shape of the contour's bounding box:
        canvas = np.zeros((self.box.height, self.box.width), dtype=np.uint8)

        # Draw contour on background canvas:
        mask = cv2.drawContours(canvas, [self.mapped_points], -1, 1, -1)

        # Convert array to boolean:
        mask = mask.astype(np.bool_)

        return mask

class projection:
    '''
    Represents a projection of the foreground pixels of an image (projection profiling).
    '''

    def __init__(self, signal):
        '''
        Constructs a projection object from an already extracted signal.
        '''
        self.signal = signal

    @classmethod
    def from_image(image, axis, sigma=3):
        '''
        Constructs a projection object from the signal obtained by projecting the given image along given axis and 
        smoothing it through an 1D Gaussian filter with given sigma.
        '''

        # Count the foreground pixels along axis:
        signal = (image / 255).astype(np.int).sum(axis=axis)

        # Smooth the resulting signal:
        signal = gaussian_filter1d(signal, sigma)

        return projection(signal)

    def find_valleys(self):
        '''
        Retrieves the valleys (local minima) of the projection curve.
        '''

        # Get valleys of projection (peaks of negated projection):
        self.valleys, _ = find_peaks(np.negative(self.signal))

        return self.valleys

    def split_continuous_intervals(self):
        '''
        Splits projection into its continuous parts.
        '''

        # Get indices of non-zero points of the projection:
        nonzero = np.nonzero(self.signal)

        # Split consecutive indices (continuous regions) - Based on https://stackoverflow.com/a/7353335:
        consecutive = np.split(nonzero, np.where(np.diff(nonzero > 1)[0] + 1))

        projections = []
        intervals = []
        for grp in consecutive:
            if len(grp) >= 2:
                # Create a new projection from the continuous interval:
                projections.append(projection(self.signal[grp]))

                # Extract the start and end point of each consecutive interval:
                intervals.append((grp[0], grp[-1]))

        return projections, intervals

class image:
    '''
    Wrapper of cv2 image.
    '''

    def __init__(self, img):
        '''
        Constructs an image object from a cv2 binary image (foreground must be 255 (white) and background 0 (black)).
        '''

        self.img = img

    def analyse_contours(self):
        '''
        Retrieves the image contours.
        '''

        # Get contours and their respective hierarchy information:
        polygons, hierarchy = cv2.findContours(self.img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Remove redundant axis 0:
        hierarchy = hierarchy.reshape(hierarchy.shape[1], hierarchy.shape[2])

        self.contours = []
        for polygon, hier in zip(polygons, hierarchy):
            # Remove redundant axis 1:
            polygon = polygon.reshape(polygon.shape[0], polygon.shape[2])

            # Save polygon and hierarchy information as a contour object:
            self.contours.append(contour(polygon, hier))

        return self.contours

    def crop(self, box):
        '''
        Crops the image given a bounding box.
        '''

        return image(self.img[box.y0:box.y1, box.x0:box.x1])

    def mask(self, msk):
        '''
        Masks the image by setting False part to background.
        '''

        self.img[not msk] = 0

    def reshape(self, shape):
        '''
        Reshapes the image to given shape (np.reshape wrapper).
        '''

        self.img = self.img.reshape(shape)

        return self.img

    def resize(self, shape):
        '''
        Resizes the image to given shape (cv2.resize wrapper).
        '''

        self.img = cv2.resize(self.img, shape, interpolation=cv2.INTER_NEAREST)

        return self.img
