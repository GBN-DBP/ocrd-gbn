import numpy as np
import cv2
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

class BoundingBox:
    '''
    Represents the bounding box of a shape in an image.
    '''

    def __init__(self, rectangle):
        '''
        Constructs a BoundingBox object from a cv2 rectangle (x, y, w, h).
        '''

        # Parse the rectangle:
        self.x0, self.y0, self.width, self.height = rectangle

        # Calculate boundaries on each axis:
        self.x1 = self.x0 + self.width
        self.y1 = self.y0 + self.height

        # Calculate area of bounding box:
        self.area = self.width * self.height

    @classmethod
    def from_polygon(self, polygon):
        '''
        Constructs a BoundingBox object from a Polygon object through cv2.boundingRect.
        '''

        return BoundingBox(cv2.boundingRect(polygon.points))

    def split(self, intervals, axis):
        '''
        Splits the bounding box along given axis given a list of intervals. An interval consist of a tuple (start, end)
        representing the interval [start, end[ along the axis to be split.
        '''

        boxes = []
        if axis:
            # On y-axis (1):
            for y0, y1 in intervals:
                boxes.append(BoundingBox([self.x0, y0, self.width, y1 - y0]))
        else:
            # on x-axis (0):
            for x0, x1 in intervals:
                boxes.append(BoundingBox([x0, self.y0, x1 - x0, self.height]))

        return boxes

class Polygon:
    '''
    Represents a polygon.
    '''

    def __init__(self, points):
        '''
        Constructs a Polygon object from a list of points.
        '''

        self.points = points

        # Extract bounding box of polygon:
        self.bbox = BoundingBox.from_polygon(self)

        # Map points to origin (x0 == 0, y0 == 0):
        self.mapped_points = np.stack((self.points[:, 0] - self.bbox.x0, self.points[:, 1] - self.bbox.y0), axis=1)

    def is_valid(self):
        '''
        Checks the validity of the polygon.
        '''

        return len(self.points) >= 3

    def to_mask(self):
        '''
        Converts polygon to mask.
        '''

        # Create a background canvas with the shape of the polygon's bounding box:
        canvas = np.zeros((self.bbox.height, self.bbox.width), dtype=np.uint8)

        # Draw polygon on background canvas:
        mask = cv2.fillPoly(canvas, np.int32([self.mapped_points]), 1)

        # Convert array to boolean:
        mask = mask.astype(np.bool_)

        return mask

class Contour:
    '''
    Wrapper of cv2 contour (shape of an image).
    '''

    def __init__(self, contour, hierarchy):
        '''
        Constructs a Contour object. Both the actual contour and the hierarchy returned from the cv2.findContours call
        with mode cv2.RETR_TREE must be provided.
        '''

        self.contour = contour

        # Get area of contour:
        self.area = cv2.contourArea(self.contour)

        # Parse hierarchy from cv2 array:
        self.next, self.previous, self.first_child, self.parent = hierarchy

        # Remove redundant axis 1 and wrap contour in Polygon object:
        self.polygon = Polygon(self.contour.reshape(self.contour.shape[0], self.contour.shape[2]))

    @classmethod
    def from_image(self, image):
        '''
        Retrieves the image contours and wraps them in Contour objects.
        '''

        # Get contours and their respective hierarchy information:
        contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if contours: 
            # Remove redundant axis 0:
            hierarchy = hierarchy.reshape(hierarchy.shape[1], hierarchy.shape[2])

            instances = []
            for cnt, h in zip(contours, hierarchy):
                # Wrap contour and hierarchy info in Contour object::
                instances.append(Contour(cnt, h))

            return instances
        else:
            return []

    def is_child(self):
        '''
        Checks whether the contour is a child of another contour along the hierarchy.
        '''

        return self.parent != -1

class Projection:
    '''
    Represents a projection of the foreground pixels of an image (for projection profiling).
    '''

    def __init__(self, signal):
        '''
        Constructs a Projection object from an already extracted signal.
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

        return Projection(signal)

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

        intervals = []
        for grp in consecutive:
            if len(grp) >= 2:
                # Extract the start and end point of each consecutive interval:
                intervals.append((grp[0], grp[-1]))

        return intervals
