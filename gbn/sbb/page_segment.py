from gbn.tool import OCRD_TOOL

from ocrd import Processor
from ocrd_modelfactory import page_from_file
from ocrd_models.ocrd_page import to_xml
from ocrd_models.ocrd_page_generateds import AlternativeImageType, CoordsType, LabelsType, LabelType, MetadataItemType, TextRegionType
from ocrd_utils import concat_padded, coordinates_for_segment, getLogger, MIMETYPE_PAGE, points_from_polygon

import os.path
import cv2
import numpy as np
import PIL.Image
from shapely import geometry

TOOL = "ocrd-gbn-sbb-page-segment"

LOG = getLogger("processor.PageSegment")
FALLBACK_FILEGRP_IMG = "OCR-D-IMG-SEG" # DEBUG ONLY

class PageSegment(Processor):
    def __init__(self, *args, **kwargs):
        kwargs['ocrd_tool'] = OCRD_TOOL['tools'][TOOL]
        kwargs['version'] = OCRD_TOOL['version']
        super(PageSegment, self).__init__(*args, **kwargs)

        if hasattr(self, "input_file_grp"):
            try:
                self.input_file_grp, self.txtreg_grp, self.txtline_grp = self.input_file_grp.split(',')
            except ValueError:
                LOG.error("Three input groups required (input_group, txtreg_group, txtline_group)")
                quit()

            try:
                self.page_grp, self.image_grp = self.output_file_grp.split(',')
            except ValueError:
                self.page_grp = self.output_file_grp
                self.image_grp = FALLBACK_FILEGRP_IMG # DEBUG ONLY
                LOG.info("No output file group for images specified, falling back to '%s'", FALLBACK_FILEGRP_IMG)

    @property
    def txtreg_files(self):
        """
        List the text region prediction image files
        """
        return self.workspace.mets.find_files(fileGrp=self.txtreg_grp, pageId=self.page_id)

    @property
    def txtline_files(self):
        """
        List the text line prediction image files
        """
        return self.workspace.mets.find_files(fileGrp=self.txtline_grp, pageId=self.page_id)

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
        predict_image = cv2.erode(predict_image, self.kernel, iterations=4)
        predict_image = cv2.dilate(predict_image, self.kernel, iterations=4)

        return cv2.findContours(predict_image, cv2.cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    def extract_polygons(self, contours):
        polygons = []

        for contour in contours:
            polygons.append(geometry.Polygon([point[0] for point in contour]))

        return polygons

    def extract_boxes(self, polygons):
        boxes = []
        
        for polygon in polygons:
            x, y, w, h = cv2.boundingRect(np.array([[point] for point in polygon.exterior.coords], dtype=np.uint))
            boxes.append([x, y, w, h])

        return boxes

    def process(self):
        for n, (input_file, txtreg_file, txtline_file) in enumerate(zip(self.input_files, self.txtreg_files, self.txtline_files)):
            LOG.info("Processing binary page image input file %i / %s", n, input_file)
            LOG.info("Processing text regions prediction image input file %i / %s", n, txtreg_file)
            LOG.info("Processing text lines prediction image input file %i / %s", n, txtline_file)

            # Create a new PAGE file from the input file:
            page_id = input_file.pageId or input_file.ID
            pcgts = page_from_file(self.workspace.download_file(input_file))
            page = pcgts.get_Page()

            # Get image from PAGE (must have been binarized):
            page_image, page_xywh, _ = self.workspace.image_from_page(
                page,
                page_id,
                feature_selector="binarized"
            )

            # Convert PIL image array to 8-bit grayscale then to uint8 Numpy array (for OpenCV):
            page_image = np.array(page_image.convert('L'), dtype=np.uint8)

            # Create a new PAGE file from text region prediction image file:
            txtreg_page_id = txtreg_file.pageId or txtreg_file.ID
            txtreg_pcgts = page_from_file(self.workspace.download_file(txtreg_file))
            txtreg_page = txtreg_pcgts.get_Page()

            # Get text regions prediction image from PAGE:
            txtreg, _, _ = self.workspace.image_from_page(
                txtreg_page,
                txtreg_page_id,
            )

            # Convert PIL image array to 8-bit grayscale then to uint8 Numpy array (for OpenCV):
            txtreg = np.array(txtreg.convert('L'), dtype=np.uint8)

            # Create a new PAGE file from the text line prediction image file:
            txtline_page_id = txtline_file.pageId or txtline_file.ID
            txtline_pcgts = page_from_file(self.workspace.download_file(txtline_file))
            txtline_page = txtline_pcgts.get_Page()

            # Get text lines prediction image from PAGE:
            txtline, _, _ = self.workspace.image_from_page(
                txtline_page,
                txtline_page_id,
            )

            # Convert PIL image array to 8-bit grayscale then to uint8 Numpy array (for OpenCV):
            txtline = np.array(txtline.convert('L'), dtype=np.uint8)

            # TODO: Unhardcode this:
            self.kernel = np.ones((5, 5), np.uint8)

            # Retrieve contours of segmented regions:
            contours, hierarchy = self.extract_contours(txtreg)

            # Filter out non-polygons:
            contours = self.geometry_filter(contours)

            # Filter out non-leaves (contours with child contours):
            contours = self.hierarchy_filter(contours, hierarchy)

            polygons = self.extract_polygons(contours)

            # Filter out small/big particles:
            area = page_image.shape[0] * page_image.shape[1]
            contours = self.particle_size_filter(
                polygons,
                self.parameter['min_particle_size'] * area,
                self.parameter['max_particle_size'] * area
            )

            boxes = self.extract_boxes(polygons)

            # Filter out regions with big/small textline density (white foreground):
            boxes = self.foreground_density_filter(
                txtline,
                boxes,
                self.parameter['min_textline_density'],
                self.parameter['max_textline_density'],
                255
            )

            # Filter out regions with big/small foreground density (black foreground):
            boxes = self.foreground_density_filter(
                page_image,
                boxes,
                self.parameter['min_foreground_density'],
                self.parameter['max_foreground_density'],
                0
            )

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
