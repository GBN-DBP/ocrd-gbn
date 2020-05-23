from gbn.tool import OCRD_TOOL

from ocrd import Processor
from ocrd_modelfactory import page_from_file
from ocrd_models.ocrd_page import to_xml
from ocrd_models.ocrd_page_generateds import AlternativeImageType, CoordsType, LabelsType, LabelType, MetadataItemType, TextRegionType
from ocrd_utils import concat_padded, coordinates_for_segment, getLogger, MIMETYPE_PAGE, points_from_polygon

import os.path
import cv2
import math
import numpy as np
import PIL.Image
import shapely.geometry
import scipy.signal
import scipy.ndimage
import matplotlib.pyplot as plt

TOOL = "ocrd-gbn-sbb-region-segment"

LOG = getLogger("processor.RegionSegment")
FALLBACK_FILEGRP_IMG = "OCR-D-IMG-SEG" # DEBUG ONLY

class RegionSegment(Processor):
    def __init__(self, *args, **kwargs):
        kwargs['ocrd_tool'] = OCRD_TOOL['tools'][TOOL]
        kwargs['version'] = OCRD_TOOL['version']
        super(RegionSegment, self).__init__(*args, **kwargs)

        if hasattr(self, "output_file_grp"):
            try:
                self.page_grp, self.image_grp = self.output_file_grp.split(',')
            except ValueError:
                self.page_grp = self.output_file_grp
                self.image_grp = FALLBACK_FILEGRP_IMG # DEBUG ONLY
                LOG.info("No output file group for images specified, falling back to '%s'", FALLBACK_FILEGRP_IMG)

    def geometry_filter(self, contours):
        '''
        Filters out contours that do not compose valid polygons
        '''
        filtered = []

        for contour in contours:
            # Polygons must have at least 3 vertices:
            if len(contour) >= 3:
                filtered.append(contour)

        return filtered

    def hierarchy_filter(self, contours, hierarchy):
        '''
        Filters out contours that contain inner contours (child contours)
        '''
        filtered = []

        for idx in range(len(contours)):
            # If 'First Child' field of contours does not point to another contour:
            if hierarchy[0][idx][3] == -1:
                filtered.append(contours[idx])

        return filtered

    def particle_size_filter(self, polygons, min_area, max_area):
        '''
        Filters out polygons whose area is too small or too big
        '''
        filtered = []

        for polygon in polygons:
            if polygon.area >= min_area and polygon.area <= max_area:
                filtered.append(polygon)

        return filtered

    def foreground_density_filter(self, img, boxes, min_density, max_density, fg_color):
        '''
        Filters out boxes that when sliced from the given image generate chunks with too small or too big foreground density
        '''
        filtered = []

        for box in boxes:
            # Slice chunk from image:
            chunk = img[box[1]:box[1]+box[3], box[0]:box[0]+box[2]]

            total_pixels = chunk.shape[0] * chunk.shape[1]
            fg_pixels = np.count_nonzero(chunk == fg_color)

            density = fg_pixels / total_pixels

            if density >= min_density and density <= max_density:
                filtered.append(box)

        return filtered

    def extract_contours(self, predict_image):
        # Smooth shapes:
        #predict_image = cv2.erode(predict_image, self.kernel, iterations=2)
        #predict_image = cv2.dilate(predict_image, self.kernel, iterations=2)

        return cv2.findContours(predict_image, cv2.cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    def extract_polygons(self, contours):
        polygons = []

        for contour in contours:
            polygons.append(shapely.geometry.Polygon([point[0] for point in contour]))

        return polygons

    def extract_boxes(self, polygons):
        boxes = []
        
        for polygon in polygons:
            x, y, w, h = cv2.boundingRect(np.array([[point] for point in polygon.exterior.coords], dtype=np.uint))
            boxes.append([x, y, w, h])

        return boxes

    def process(self):
        for n, input_file in enumerate(self.input_files):
            LOG.info("Processing page image input file %i / %s", n, input_file)

            # Create a new PAGE file from the input file:
            page_id = input_file.pageId or input_file.ID
            pcgts = page_from_file(self.workspace.download_file(input_file))
            page = pcgts.get_Page()

            # Get image from PAGE:
            page_image, page_xywh, _ = self.workspace.image_from_page(
                page,
                page_id,
            )

            regions = page.get_TextRegion()

            # DEBUG
            page_image_cv2 = np.array(page_image.convert('L'), dtype=np.uint8)
            page_mask = (page_image_cv2 / 255).astype(np.bool_)
            page_image_cv2[page_mask == True] = 0
            page_image_cv2[page_mask == False] = 255
            page_image_cv2 = cv2.cvtColor(page_image_cv2, cv2.COLOR_GRAY2BGR)
            #######

            # TODO: Unhardcode this:
            self.kernel = np.ones((5, 5), np.uint8)

            #homogen = []
            #non_homogen = []

            for idx, region in enumerate(regions):
                # Get image from text region:
                region_image, _ = self.workspace.image_from_segment(region, page_image, page_xywh)

                # Remove alpha channel from image, if there is one:
                if region_image.mode == 'LA' or region_image.mode == 'RGBA':
                    # Ensure LA:
                    region_image = region_image.convert('LA')

                    alpha = region_image.getchannel('A')

                    # Paste image on a black canvas:
                    canvas = PIL.Image.new('LA', region_image.size, 0)
                    canvas.paste(region_image, mask=alpha)

                    region_image = canvas
                else:
                    alpha = None

                # Convert PIL image array to 8-bit grayscale then to uint8 Numpy array (for OpenCV):
                region_image = np.array(region_image.convert('L'), dtype=np.uint8)

                # Filter out regions with big/small textline density (white foreground):
                boxes = self.foreground_density_filter(
                    region_image,
                    [[0, 0, region_image.shape[1], region_image.shape[0]]],
                    self.parameter['min_foreground_density'],
                    self.parameter['max_foreground_density'],
                    255
                )

                if boxes:
                    coords = region.get_Coords()
                
                    x_offsets = coords.get_points().replace(' ', ',').split(',')[0::2]
                    y_offsets = coords.get_points().replace(' ', ',').split(',')[1::2]

                    reg_x0 = int(x_offsets[0])
                    reg_y0 = int(y_offsets[0])

                    cx = region_image.shape[1] / 2
                    cy = region_image.shape[0] / 2

                    angle = region.get_orientation()

                    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)

                    #non_homogen.append((region_image, reg_x0, reg_y0, M))

            #while non_homogen:
                #new_non_homogen = []

                #for idx, item in enumerate(non_homogen):
                    #region_image, reg_x0, reg_y0, M = item

                    #peak_signal = (region_image / 255).astype(np.uint8).sum(axis=1)
                    #peak_signal = scipy.ndimage.gaussian_filter1d(peak_signal, 3)
                    #peak_signal = np.gradient(peak_signal)

                    #peaks, _ = scipy.signal.find_peaks(peak_signal, height=0)

                    #if peaks.shape[0] > 1:
                        #left_peaks = peaks[:-1]
                        #right_peaks = peaks[1:]

                        #diffs = np.subtract(right_peaks, left_peaks)

                        #if np.var(diffs) < 4.25:
                            #homogen.append((region_image, reg_x0, reg_y0, M))
                        #else:
                            #mask = (region_image / 255).astype(np.bool_)

                            #inv_region_image = np.ones_like(region_image)

                            #inv_region_image[mask == True] = 0

                            #valley_signal = inv_region_image.astype(np.uint8).sum(axis=1)
                            #valley_signal = scipy.ndimage.gaussian_filter1d(valley_signal, 3)

                            #valleys, _ = scipy.signal.find_peaks(valley_signal, height=0)

                            #valleys_density = inv_region_image[valleys].sum(axis=1)
                            #valley = valleys[np.argmax(valleys_density)]

                            #new_non_homogen.append((region_image[:valley], reg_x0, reg_y0, M))
                            #new_non_homogen.append((region_image[valley:], reg_x0, reg_y0 + valley, M))
                    #else:
                        #homogen.append((region_image, reg_x0, reg_y0, M))
                
                #non_homogen = new_non_homogen

            #for item in homogen:
                #region_image, reg_x0, reg_y0, M = item

                    mask = (region_image / 255).astype(np.bool_)

                    inv_region_image = np.ones_like(region_image)

                    inv_region_image[mask == True] = 0

                    valley_signal = inv_region_image.astype(np.uint8).sum(axis=1)
                    valley_signal = scipy.ndimage.gaussian_filter1d(valley_signal, 3)

                    valleys, _ = scipy.signal.find_peaks(valley_signal, height=0)

                    x0 = 0
                    x1 = region_image.shape[1] - 1
                    y0 = 0
                    y1 = 0

                    points = np.hstack((x0,y0,x1,y1)).reshape(-1, 2)
                    points = np.hstack((points, np.ones((points.shape[0], 1), dtype=np.uint8)))

                    points = np.dot(M, points.T).T

                    x0 = reg_x0 + int(points[0, 0])
                    y0 = reg_y0 + int(points[0, 1])
                    x1 = reg_x0 + int(points[1, 0])
                    y1 = reg_y0 + int(points[1, 1])

                    cv2.line(page_image_cv2, (x0, y0), (x1, y1), (0, 255, 0), 2)

                    x0 = 0
                    x1 = 0
                    y0 = 0
                    y1 = region_image.shape[0] - 1

                    points = np.hstack((x0,y0,x1,y1)).reshape(-1, 2)
                    points = np.hstack((points, np.ones((points.shape[0], 1), dtype=np.uint8)))

                    points = np.dot(M, points.T).T

                    x0 = reg_x0 + int(points[0, 0])
                    y0 = reg_y0 + int(points[0, 1])
                    x1 = reg_x0 + int(points[1, 0])
                    y1 = reg_y0 + int(points[1, 1])

                    cv2.line(page_image_cv2, (x0, y0), (x1, y1), (0, 255, 0), 2)

                    x0 = region_image.shape[1] - 1
                    x1 = region_image.shape[1] - 1
                    y0 = 0
                    y1 = region_image.shape[0] - 1

                    points = np.hstack((x0,y0,x1,y1)).reshape(-1, 2)
                    points = np.hstack((points, np.ones((points.shape[0], 1), dtype=np.uint8)))

                    points = np.dot(M, points.T).T

                    x0 = reg_x0 + int(points[0, 0])
                    y0 = reg_y0 + int(points[0, 1])
                    x1 = reg_x0 + int(points[1, 0])
                    y1 = reg_y0 + int(points[1, 1])

                    cv2.line(page_image_cv2, (x0, y0), (x1, y1), (0, 255, 0), 2)

                    x0 = 0
                    x1 = region_image.shape[1] - 1
                    y0 = region_image.shape[0] - 1
                    y1 = region_image.shape[0] - 1

                    points = np.hstack((x0,y0,x1,y1)).reshape(-1, 2)
                    points = np.hstack((points, np.ones((points.shape[0], 1), dtype=np.uint8)))

                    points = np.dot(M, points.T).T

                    x0 = reg_x0 + int(points[0, 0])
                    y0 = reg_y0 + int(points[0, 1])
                    x1 = reg_x0 + int(points[1, 0])
                    y1 = reg_y0 + int(points[1, 1])

                    cv2.line(page_image_cv2, (x0, y0), (x1, y1), (0, 255, 0), 2)

                    for valley in valleys:
                        valley_density = (region_image[valley] / 255).astype(np.uint8).sum(axis=0) / region_image.shape[1]

                        if valley_density < 0.5:
                            x0 = 0
                            x1 = region_image.shape[1] - 1
                            y0 = valley
                            y1 = valley

                            points = np.hstack((x0,y0,x1,y1)).reshape(-1, 2)
                            points = np.hstack((points, np.ones((points.shape[0], 1), dtype=np.uint8)))

                            points = np.dot(M, points.T).T

                            x0 = reg_x0 + int(points[0, 0])
                            y0 = reg_y0 + int(points[0, 1])
                            x1 = reg_x0 + int(points[1, 0])
                            y1 = reg_y0 + int(points[1, 1])

                            cv2.line(page_image_cv2, (x0, y0), (x1, y1), (0, 255, 0), 2)

                # Retrieve contours of segmented lines:
                #contours, hierarchy = self.extract_contours(region_image)

                # Filter out non-polygons:
                #contours = self.geometry_filter(contours)

                # Filter out non-leaves (contours with child contours):
                #contours = self.hierarchy_filter(contours, hierarchy)

                #polygons = self.extract_polygons(contours)

                # Filter out small/big particles:
                #area = region_image.shape[0] * region_image.shape[1]
                #contours = self.particle_size_filter(
                    #polygons,
                    #self.parameter['min_particle_size'] * area,
                    #self.parameter['max_particle_size'] * area
                #)

                #boxes = self.extract_boxes(polygons)

                #coords = region.get_Coords()
                
                #x_offsets = coords.get_points().replace(' ', ',').split(',')[0::2]
                #y_offsets = coords.get_points().replace(' ', ',').split(',')[1::2]

                #reg_x0 = int(x_offsets[0])
                #reg_x1 = int(x_offsets[1])
                #reg_y0 = int(y_offsets[0])
                #reg_y1 = int(y_offsets[2])

                #for line, box in enumerate(boxes):
                    #x0 = reg_x0 + box[0]
                    #y0 = reg_y0 + box[1]
                    #x1 = reg_x0 + box[0] + box[2]
                    #y1 = reg_y0 + box[1] + box[3]

                    #polygon = [
                        #[x0, y0],
                        #[x1, y0],
                        #[x1, y1],
                        #[x0, y1]
                    #]

                    #cv2.rectangle(page_image_cv2, (x0, y0), (x1, y1), (0, 255, 0), 2)
                ############

            # DEBUG ONLY
            #if alpha:
                # Convert OpenCV image array (Numpy) to PIL image array then to 8-bit grayscale with alpha channel:
                #region_image = PIL.Image.fromarray(cv2.cvtColor(region_image, cv2.COLOR_BGR2RGB))
                #region_image.putalpha(alpha)
            #else:
                # Convert OpenCV image array (Numpy) to PIL image array then to 1-bit grayscale:
                #region_image = PIL.Image.fromarray(cv2.cvtColor(region_image, cv2.COLOR_BGR2RGB))

            page_image = PIL.Image.fromarray(cv2.cvtColor(page_image_cv2, cv2.COLOR_BGR2RGB))

            # Get file ID of image to be saved:
            file_id = input_file.ID.replace(self.input_file_grp, self.image_grp)

            if file_id == input_file.ID:
                file_id = concat_padded(self.image_grp, n)

            # Concatenate region number and model name to ID:
            #file_id += "_region%04d" % idx

            # Save image:
            file_path = self.workspace.save_image_file(
                page_image,
                file_id,
                page_id=page_id,
                file_grp=self.image_grp
            )

            continue

            # DEBUG ONLY
            txtline_mask = (txtline / 255).astype(np.bool_)
            bg_mask = (page_image / 255).astype(np.bool_)

            page_image = cv2.cvtColor(page_image, cv2.COLOR_GRAY2BGR)

            page_image[bg_mask == True] = np.array([0, 0, 0])
            page_image[txtline_mask == True] = np.array([255, 0, 0])
            page_image[bg_mask == False] = np.array([255, 255, 255])
            ############

            for region, box in enumerate(boxes):
                x0 = box[0]
                y0 = box[1]
                x1 = box[0] + box[2]
                y1 = box[1] + box[3]

                polygon = [
                    [x0, y0],
                    [x1, y0],
                    [x1, y1],
                    [x0, y1]
                ]

                # DEBUG ONLY
                cv2.rectangle(page_image, (x0, y0), (x1, y1), (0, 255, 0), 3)
                ############

                region_id = page_id + "_region%04d" % region

                # convert back to absolute (page) coordinates:
                region_polygon = coordinates_for_segment(polygon, page_image, page_xywh)

                # annotate result:
                page.add_TextRegion(
                    TextRegionType(
                        id=region_id,
                        Coords=CoordsType(
                            points=points_from_polygon(region_polygon)
                        )
                    )
                )

            # DEBUG ONLY
            page_image = PIL.Image.fromarray(cv2.cvtColor(page_image, cv2.COLOR_BGR2RGB))

            file_id = txtline_file.ID.replace(self.input_file_grp, self.image_grp)

            if file_id == txtline_file.ID:
                file_id = concat_padded(self.image_grp, n)

            self.workspace.save_image_file(
                page_image,
                file_id,
                page_id=page_id,
                file_grp=self.image_grp
            )
            ############

            # Add metadata about this operation:
            metadata = pcgts.get_Metadata()
            metadata.add_MetadataItem(
                MetadataItemType(
                    type_="processingStep",
                    name=self.ocrd_tool['steps'][0],
                    value=TOOL,
                    Labels=[
                        LabelsType(
                            externalModel="ocrd-tool",
                            externalId="parameters",
                            Label=[
                                LabelType(
                                    type_=name,
                                    value=self.parameter[name]
                                ) for name in self.parameter.keys()
                            ]
                        )
                    ]
                )
            )

            # Get file ID of XML PAGE to be saved:
            file_id = input_file.ID.replace(self.input_file_grp, self.page_grp)

            if file_id == txtline_file.ID:
                file_id = concat_padded(self.page_grp, n)

            # Save XML PAGE:
            self.workspace.add_file(
                 ID=file_id,
                 file_grp=self.output_file_grp,
                 pageId=page_id,
                 mimetype=MIMETYPE_PAGE,
                 local_filename=os.path.join(self.output_file_grp, file_id)+".xml",
                 content=to_xml(pcgts)
            )
