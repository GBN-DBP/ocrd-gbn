from gbn.lib.util import resolve_box, slice_image, invert_image

import numpy as np
import cv2
import scipy.ndimage
import scipy.signal

def get_box_area(box):
    '''
    Calculates the are of given box
    '''

    # width * height:
    return box[2] * box[3]

def overlap(a, b):
    '''
    Checks if there is an overlap between given boxes
    '''

    ax0, ay0, ax1, ay1 = resolve_box(a)
    bx0, by0, bx1, by1 = resolve_box(b)

    # Check if there is an 1-d overlap for the coordinates of each axis:
    x_overlap = (ax0 <= bx0 and bx0 < ax1) or (ax0 < bx1 and bx1 <= ax1) or (bx0 <= ax0 and ax0 < bx1) or (bx0 < ax1 and ax1 <= bx1)
    y_overlap = (ay0 <= by0 and by0 < ay1) or (ay0 < by1 and by1 <= ay1) or (by0 <= ay0 and ay0 < by1) or (by0 < ay1 and ay1 <= by1)

    # If there are overlaps on both axis, both boxes overlap each other:
    return x_overlap and y_overlap

def merge_boxes(boxes):
    '''
    Merges the given boxes into a single bounding box
    '''

    # Store resolved boxes as a Numpy array:
    boxes = np.array([resolve_box(box) for box in boxes])

    # Get extremes:
    x0 = np.amin(boxes[:, 0])
    y0 = np.amin(boxes[:, 1])
    x1 = np.amax(boxes[:, 2])
    y1 = np.amax(boxes[:, 3])

    return [x0, y0, x1 - x0, y1 - y0]

def project_foreground(image, axis=0, fg_color=255):
    '''
    Projects the foreground pixels of the given image along the given axis
    '''
    # Ensure foreground is nonzero and background zero:
    if fg_color == 0:
        image = invert_image(image)

    # Count the foreground pixels along axis:
    proj = (image / 255).astype(np.int).sum(axis=axis)

    # Smooth the resulting signal:
    proj = scipy.ndimage.gaussian_filter1d(proj, 3)

    return proj

def find_valleys(projection):
    '''
    Finds the local minimum (valleys) of given projection
    '''

    # Get valleys of projection (peaks of negated projection):
    valleys, _ = scipy.signal.find_peaks(np.negative(projection))

    return valleys

def split_continuous_parts(proj):
    '''
    Splits the given foreground projection into continuous projections
    '''
    # If whole projection is already continuous:
    if all(proj) != 0:
        return [(0, proj.shape[0] - 1)]

    parts = []

    i0 = None
    for i in range(proj.shape[0]):
        # If got non zero and there was no continuous part being marked:
        if i0 is None and proj[i] > 0:
            # Mark start point of continuous part:
            i0 = i
        # If got zero and there was a continuous part being marked:
        elif i0 is not None and proj[i] == 0:
            # Mark end point of continuous part:
            i1 = i

            # Append part:
            parts.append((i0, i1))

            # Reset markers:
            i0 = None
            i1 = None

    # If there was a continuous part being marked:
    if i0 is not None:
        # Mark end point of continuous part:
        i1 = proj.shape[0]

        # Append part:
        parts.append((i0, i1))

    return parts

def split_boxes(box, parts, axis=0):
    '''
    Splits given box along given axis given a list of parts (tuples representing the intervals of the new boxes along axis)
    '''

    x_offset, y_offset, x1, y1 = resolve_box(box)

    split = []
    if axis:
        for y0, y1 in parts:
            split.append([x_offset, y_offset + y0, x1 - x_offset, y1 - y0])
    else:
        for x0, x1 in parts:
            split.append([x_offset + x0, y_offset, x1 - x0, y1 - y_offset])

    return split

def get_outliers(data, n=2):
    '''
    Retrieves indices outlier elements of given data (more than n standard deviations away from the mean)
    '''

    return np.where((abs(data - np.mean(data)) > n * np.std(data)) == True)

class Extracting():
    '''
    Methods for extracting characteristics of images through contour analysis/manipulation
    '''
    def __init__(self, image, fg_color=255, contour_area_filter=(0, 1), fg_density_filter=(0., 1.)):
        # Store image to be processed:
        self.image = image

        # Store filter parameters:
        self.min_area, self.max_area = contour_area_filter
        self.min_density, self.max_density = fg_density_filter

        # Get dimensions of image:
        self.h, self.w = image.shape

        # Get total area of image:
        self.image_area = self.w * self.h

        # Ensure foreground color is white:
        if fg_color == 0:
            self.image = invert_image(image)

        # Project the foreground pixels of the given image along both axis:
        self.projections = (
            project_foreground(self.image, axis=0),
            project_foreground(self.image, axis=1)
        )

        # Kernel for erosion/dilation:
        self.kernel = np.ones((5, 5), np.uint8)

    def erode(self, iterations):
        '''
        Erodes contours of given image
        '''

        self.image = cv2.erode(self.image, self.kernel, iterations=iterations)

    def dilate(self, iterations):
        '''
        Dilates contours of given image
        '''

        self.image = cv2.dilate(self.image, self.kernel, iterations=iterations)

    def analyse_contours(self, sub_image=None):
        '''
        Extract contours and their respective boxes
        '''

        if sub_image is None:
            image = self.image
        else:
            image = sub_image

        self.contours, self.hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        self.boxes = np.array([cv2.boundingRect(contour) for contour in self.contours])
        self.resolved_boxes = np.array([resolve_box(box) for box in self.boxes])

    def filter_boxes(self, indices):
        '''
        Updates the lists of boxes given a list of indices
        '''

        self.boxes = self.boxes[indices]
        self.resolved_boxes = self.resolved_boxes[indices]

    def filter_contours(self, indices):
        '''
        Updates the lists of contours and boxes given a list of indices
        '''

        self.contours = [self.contours[idx] for idx in indices]
        self.filter_boxes(indices)

    def filter_by_hierarchy(self):
        '''
        Filters out contours that contain inner contours (child contours)
        '''

        # Filter out all contours that have pointers to children:
        indices = [idx for idx, _ in enumerate(self.contours) if self.hierarchy[0][idx][3] == -1]

        # Update contours and boxes:
        self.filter_contours(indices)

    def filter_by_geometry(self):
        '''
        Filters out contours that do not compose valid polygons
        '''

        # Filter out all contours with less then 3 points:
        indices = [idx for idx, contour in enumerate(self.contours) if len(contour) >= 3]

        # Update contours and boxes:
        self.filter_contours(indices)

    def filter_by_area(self):
        '''
        Filters out contours whose areas are too small or too big
        '''

        indices = []
        for idx, contour in enumerate(self.contours):
            # Get area of contour:
            area = cv2.contourArea(contour)

            if area >= self.min_area and area <= self.max_area:
                indices.append(idx)

        # Update contours and boxes:
        self.filter_contours(indices)

    def filter_by_foreground_density(self):
        '''
        Filters out bounding boxes of contours that have too big or too small foreground density
        '''

        indices = []
        for idx, box in enumerate(self.boxes):
            # Get foreground density of image:
            density = self.get_foreground_density(slice_image(self.image, box))

            if density >= self.min_density and density <= self.max_density:
                indices.append(idx)

        # Update boxes:
        self.filter_boxes(indices)

    def get_foreground_density(self, sub_image=None):
        '''
        Calculates the ratio of foreground pixels of the given image
        '''

        if sub_image is None:
            image = self.image
            total_pixels = self.image_area
        else:
            image = sub_image
            total_pixels = image.shape[0] * image.shape[1]

        # Foreground color is always white (255):
        fg_pixels = np.count_nonzero(image == 255)

        if total_pixels > 0:
            return fg_pixels / total_pixels
        else:
            return 0

    def import_boxes(self, boxes):
        '''
        Imports a list of boxes into the object attribute
        '''

        self.boxes, self.resolved_boxes = boxes

        # Ensure they are Numpy arrays:
        self.boxes = np.array(self.boxes)
        self.resolved_boxes = np.array(self.resolved_boxes)

    def export_boxes(self):
        '''
        Exports the object's list of boxes
        '''

        return self.boxes, self.resolved_boxes

    def merge_overlapping_boxes(self):
        '''
        Merge all boxes wich overlap themselves
        '''

        # Sort boxes by area:
        unmerged = sorted(self.boxes, key=get_box_area, reverse=True)

        merged = []
        while unmerged:
            a = unmerged[0]
            unmerged = unmerged[1:]

            # Get all boxes which overlap with current box (a):
            overlapped = []
            for idx, b in enumerate(unmerged):
                if overlap(a, b):
                    overlapped.append(idx)

            if overlapped:
                # Split unmerged list into boxes which overlap and do not overlap the current box:
                not_overlapped = [val for idx, val in enumerate(unmerged) if idx not in overlapped]
                overlapped = [val for idx, val in enumerate(unmerged) if idx in overlapped]

                true_overlapped = []
                for box in overlapped:
                    # Store resolved boxes as a Numpy array:
                    boxes = np.array([resolve_box(a), resolve_box(box)])

                    # Get interior extremes:
                    x0 = np.amax(boxes[:, 0])
                    y0 = np.amax(boxes[:, 1])
                    x1 = np.amin(boxes[:, 2])
                    y1 = np.amin(boxes[:, 3])

                    # Intersection of both boxes:
                    intersect = [x0, y0, x1 - x0, y1 - y0]

                    # If there are foreground pixels in the intersection, it is a true overlap:
                    if np.count_nonzero(slice_image(self.image, intersect)):
                        true_overlapped.append(box)
                    else:
                        not_overlapped.append(box)

                if true_overlapped:
                    # Include current box:
                    true_overlapped.append(a)

                    # Merge all overlapping boxes:
                    merged_box = merge_boxes(true_overlapped)

                    # Include merged box here for further checking:
                    not_overlapped.append(merged_box)
                else:
                    # Include current box:
                    merged.append(a)

                # Sort boxes by area:
                unmerged = sorted(not_overlapped, key=get_box_area, reverse=True)
            else:
                merged.append(a)

        # Updates boxes:
        self.boxes = np.array(merged)
        self.resolved_boxes = np.array([resolve_box(box) for box in self.boxes])

    def merge_all_boxes(self):
        '''
        Merges all boxes into a single bounding box
        '''

        if len(self.boxes) <= 0:
            return None

        merged = merge_boxes(self.boxes)

        # Updates boxes:
        self.boxes = np.array([merged])
        self.resolved_boxes = np.array([resolve_box(box) for box in self.boxes])

        return merged

    def split_boxes_by_continuity(self, axis=0, global_projection=False):
        '''
        Splits the given boxes along given axis so that only the continuous parts of the image are bounded by them
        '''

        continuous_boxes = []
        for box in self.boxes:
            x0, y0, x1, y1 = resolve_box(box)

            if global_projection:
                # Slice from global projection:
                projection = self.projections[axis][y0:y1] if axis else self.projections[axis][x0:x1]
            else:
                # Slice from image:
                projection = project_foreground(slice_image(self.image, box), axis=axis)

            # Retrieve the continuous parts of slice of the image bounded by the box:
            continuous_boxes += split_boxes(box, split_continuous_parts(projection), axis=axis)

        # Updates boxes:
        self.boxes = np.array(continuous_boxes)
        self.resolved_boxes = np.array([resolve_box(box) for box in self.boxes])

    def split_boxes_by_standard_deviation(self, axis=0, global_projection=False, n=2):
        '''
        Splits the given boxes along given axis in the outlier valleys of the projections
        '''

        continuous_boxes = []
        for box in self.boxes:
            x0, y0, x1, y1 = resolve_box(box)

            if global_projection:
                # Slice from global projection:
                projection = self.projections[axis][y0:y1] if axis else self.projections[axis][x0:x1]
            else:
                # Slice from image:
                projection = project_foreground(slice_image(self.image, box), axis=axis)

            valleys = find_valleys(projection)

            # Get outliers:
            outlier_valleys = valleys[get_outliers(projection[valleys], n=n)]

            # Zip the start and end indices of each split part:
            parts = zip(np.append([0], outlier_valleys), np.append(outlier_valleys, [projection.shape[0]]))

            # Retrieve the continuous parts of slice of the image bounded by the box:
            continuous_boxes += split_boxes(box, parts, axis=axis)

        # Updates boxes:
        self.boxes = np.array(continuous_boxes)
        self.resolved_boxes = np.array([resolve_box(box) for box in self.boxes])

    def split_image_into_frames(self, axis=0):
        '''
        Splits the image into frames along the axis in the valleys of its projection
        '''

        valleys = find_valleys(self.projections[axis])

        # Filter valleys by foreground density:
        filtered_valleys = []
        for valley in valleys:
            # Get density:
            total_pixels = self.h if axis else self.w
            fg_pixels = self.projections[axis][valley]
            density = fg_pixels / total_pixels

            # Filter out valleys where there is more foreground than background:
            if density < 0.9:
                filtered_valleys.append(valley)

        frames = []

        if axis:
            for y0, y1 in zip(np.append([0], filtered_valleys).astype(np.int), np.append(filtered_valleys, [self.h]).astype(np.int)):
                box = [0, y0, self.w, y1 - y0]
                frames.append((slice_image(self.image, box), box))
        else:
            for x0, x1 in zip(np.append([0], filtered_valleys).astype(np.int), np.append(filtered_valleys, [self.w]).astype(np.int)):
                box = [x0, 0, x1 - x0, self.h]
                frames.append((slice_image(self.image, box), box))

        return frames
