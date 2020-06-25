from gbn.lib.util import resolve_box, pil_to_cv2_rgb, cv2_to_pil_rgb, pil_to_cv2_gray, cv2_to_pil_gray
from gbn.lib.predict import Predicting
from gbn.lib.extract import Extracting
from gbn.tool import OCRD_TOOL

from ocrd import Processor
from ocrd_modelfactory import page_from_file
from ocrd_models.ocrd_page import to_xml
from ocrd_models.ocrd_page_generateds import AlternativeImageType, CoordsType, LabelsType, LabelType, MetadataItemType, TextLineType, TextRegionType
from ocrd_utils import concat_padded, coordinates_for_segment, getLogger, MIMETYPE_PAGE, points_from_polygon

import os.path
import numpy as np
import cv2

TOOL = "ocrd-gbn-sbb-segment"

LOG = getLogger("processor.OcrdGbnSbbSegment")
FALLBACK_FILEGRP_IMG = "OCR-D-IMG-SEG"

class OcrdGbnSbbSegment(Processor):
    def __init__(self, *args, **kwargs):
        kwargs['ocrd_tool'] = OCRD_TOOL['tools'][TOOL]
        kwargs['version'] = OCRD_TOOL['version']
        super(OcrdGbnSbbSegment, self).__init__(*args, **kwargs)

        if hasattr(self, "output_file_grp"):
            try:
                self.page_grp, self.image_grp = self.output_file_grp.split(',')
                self.output_file_grp = self.page_grp
            except ValueError:
                self.page_grp = self.output_file_grp
                self.image_grp = FALLBACK_FILEGRP_IMG
                LOG.info("No output file group for images specified, falling back to '%s'", FALLBACK_FILEGRP_IMG)

    def _process_page(self, page, page_image, page_xywh, page_id, n, input_file, textregion_image, textline_image):
        # Convert PIL to cv2 (grayscale):
        page_image, alpha = pil_to_cv2_gray(page_image)

        # Construct characteristic extractor for text region prediction:
        textregion_extractor = Extracting(
            textregion_image,
            contour_area_filter=(
                self.parameter['min_particle_size'] * page.get_imageHeight() * page.get_imageWidth(),
                self.parameter['max_particle_size'] * page.get_imageHeight() * page.get_imageWidth()
            )
        )

        # Erode and dilate prediction image to separate "weakly connected" regions:
        textregion_extractor.erode(4)
        textregion_extractor.dilate(4)

        # Analyse contours and boxes:
        textregion_extractor.analyse_contours()

        # Filter out contours inside other contours:
        textregion_extractor.filter_by_hierarchy()

        # Filter out contours that do not compose valid polygons:
        textregion_extractor.filter_by_geometry()

        # Filter contours by area:
        textregion_extractor.filter_by_area()

        # Construct characteristic extractor for text line prediction:
        textline_extractor = Extracting(
            textline_image,
            fg_density_filter=(
                self.parameter['min_textline_density'],
                self.parameter['max_textline_density']
            )
        )

        # Load bounding boxes of text regions into the text line extractor:
        textline_extractor.import_boxes(textregion_extractor.export_boxes())

        # Filter boxes by textline density:
        textline_extractor.filter_by_foreground_density()

        # Merge overlapping:
        textline_extractor.merge_overlapping_boxes()

        # Segment clearly distinct regions of the page vertically using the pixels predicted as text lines:
        textline_extractor.split_boxes_by_continuity(axis=1, global_projection=True)

        # Segment separate text regions horizontally using the pixels predicted as text lines:
        textline_extractor.split_boxes_by_continuity(axis=0)
        textline_extractor.split_boxes_by_standard_deviation(axis=0, n=4)

        # Dilate text line predictions a bit to join ones that are close to each other:
        textline_extractor.dilate(2)

        # Segment separate text regions vertically using the pixels predicted as text lines:
        textline_extractor.split_boxes_by_continuity(axis=1)

        # Resize text regions horizontally using the pixels predicted as text lines:
        textline_extractor.split_boxes_by_continuity(axis=0)

        # Filter boxes by textline density:
        textline_extractor.filter_by_foreground_density()

        # Save rectangles of boxes as text regions:
        for idx, box in enumerate(textline_extractor.boxes):
            x0, y0, x1, y1 = resolve_box(box)

            polygon = [
                [x0, y0],
                [x1, y0],
                [x1, y1],
                [x0, y1]
            ]

            # Convert back to absolute (page) coordinates:
            polygon = coordinates_for_segment(polygon, page_image, page_xywh)

            region_id = page_id + "_region%04d" % idx

            # Save text region:
            page.add_TextRegion(
                TextRegionType(
                    id=region_id,
                    Coords=CoordsType(
                        points=points_from_polygon(polygon)
                    )
                )
            )

        # DEBUG ONLY
        txtline_mask = (textline_image / 255).astype(np.bool_)
        bg_mask = (page_image / 255).astype(np.bool_)

        page_image = cv2.cvtColor(page_image, cv2.COLOR_GRAY2BGR)

        page_image[bg_mask == True] = np.array([0, 0, 0])
        page_image[txtline_mask == True] = np.array([255, 0, 0])
        page_image[bg_mask == False] = np.array([255, 255, 255])

        for box in textline_extractor.boxes:
            x0, y0, x1, y1 = resolve_box(box)
            cv2.rectangle(page_image, (x0, y0), (x1, y1), (0, 255, 0), 3)

        # Convert cv2 to PIL (RGB):
        page_image = cv2_to_pil_rgb(page_image, alpha)

        # Get file ID of image to be saved:
        file_id = input_file.ID.replace(self.input_file_grp, self.image_grp)

        if file_id == input_file.ID:
            file_id = concat_padded(self.image_grp, n)

        # Save image:
        self.workspace.save_image_file(
            page_image,
            file_id,
            page_id=page_id,
            file_grp=self.image_grp
        )
        ############

    def _process_region(self, page, page_image, region, region_idx, region_image, region_xywh, page_id, n, input_file, textline_image):
        # Construct characteristic extractor for text line prediction:
        textline_extractor = Extracting(
            textline_image,
            fg_density_filter=(
                self.parameter['min_textline_density'],
                self.parameter['max_textline_density']
            )
        )

        # TODO: Achieve this through API
        # Filter whole region by textline density:
        density = textline_extractor.get_foreground_density()
        if density < self.parameter['min_textline_density'] or self.parameter['max_textline_density'] < density:
            return

        # Split region into line frames:
        frames = textline_extractor.split_image_into_frames(axis=1)

        for idx, (frame, box) in enumerate(frames):
            x_offset, y_offset, _, _ = resolve_box(box)

            # Construct characteristic extractor for text line frame:
            frame_extractor = Extracting(
                frame,
                contour_area_filter=(
                    self.parameter['min_particle_size'] * page.get_imageHeight() * page.get_imageWidth(),
                    self.parameter['max_particle_size'] * page.get_imageHeight() * page.get_imageWidth()
                ),
                fg_density_filter=(
                    self.parameter['min_textline_density'],
                    self.parameter['max_textline_density']
                )
            )

            # Dilate then erode:
            frame_extractor.dilate(2)
            frame_extractor.erode(2)

            # Analyse contours and boxes:
            frame_extractor.analyse_contours()

            # Filter out contours that do not compose valid polygons:
            frame_extractor.filter_by_geometry()

            # Filter contours by area:
            frame_extractor.filter_by_area()

            # Filter boxes by foreground:
            frame_extractor.filter_by_foreground_density()

            # Merge all boxes:
            box = frame_extractor.merge_all_boxes()

            # If no remaining boxes:
            if not box:
                return

            x0, y0, x1, y1 = resolve_box(box)

            # Apply frame offset inside region:
            x0 += x_offset
            y0 += y_offset
            x1 += x_offset
            y1 += y_offset

            polygon = [
                [x0, y0],
                [x1, y0],
                [x1, y1],
                [x0, y1]
            ]

            # Convert back to absolute (page) coordinates:
            polygon = coordinates_for_segment(polygon, region_image, region_xywh)

            region_id = page_id + "_region%04d" % region_idx + "_line%04d" % idx

            # Save text line:
            region.add_TextLine(
                TextLineType(
                    id=region_id,
                    Coords=CoordsType(
                        points=points_from_polygon(polygon)
                    )
                )
            )

            # DEBUG ONLY
            cv2.line(page_image, tuple(polygon[0]), tuple(polygon[1]), (0, 127, 0), 2)
            cv2.line(page_image, tuple(polygon[1]), tuple(polygon[2]), (0, 127, 0), 2)
            cv2.line(page_image, tuple(polygon[2]), tuple(polygon[3]), (0, 127, 0), 2)
            cv2.line(page_image, tuple(polygon[3]), tuple(polygon[0]), (0, 127, 0), 2)
            ############

    def process(self):
        if self.parameter['operation_level'] == "page":
            if os.path.isfile(self.parameter['textregion_src']):
                # Construct text region predictor:
                textregion_predictor = Predicting(self.parameter['textregion_src'], self.parameter['textregion_algorithm'])

                textregion_images = []
                for (n, input_file) in enumerate(self.input_files):
                    LOG.info("Predicting text regions of input file: %i / %s", n, input_file)

                    # Create a new PAGE file from the input file:
                    page_id = input_file.pageId or input_file.ID
                    pcgts = page_from_file(self.workspace.download_file(input_file))
                    page = pcgts.get_Page()

                    # Get binarized and deskewed image from PAGE:
                    page_image, page_xywh, _ = self.workspace.image_from_page(
                        page,
                        page_id,
                        feature_selector="binarized,deskewed"
                    )

                    # Convert PIL to cv2 (RGB):
                    page_image, _ = pil_to_cv2_rgb(page_image)

                    # Get labels per-pixel and map them to grayscale:
                    textregion_images.append(textregion_predictor.predict(page_image) * 255)

                # Close session for text regions:
                textregion_predictor.session.close()
            else:
                # Get prediction images from given file group:
                textregion_input_files = self.workspace.mets.find_files(
                    fileGrp=self.parameter['textregion_src']
                )

                textregion_images = []
                for (n, input_file) in enumerate(textregion_input_files):
                    LOG.info("Retrieving cached text region prediction of input file: %i / %s", n, input_file)

                    # Create a new PAGE file from the input file:
                    page_id = input_file.pageId or input_file.ID
                    pcgts = page_from_file(self.workspace.download_file(input_file))
                    page = pcgts.get_Page()

                    # Get binarized and deskewed image from PAGE:
                    page_image, page_xywh, _ = self.workspace.image_from_page(
                        page,
                        page_id
                    )

                    # Convert PIL to cv2 (gray):
                    page_image, _ = pil_to_cv2_gray(page_image, bg_color=0)

                    textregion_images.append(page_image)

            if os.path.isfile(self.parameter['textline_src']):
                # Construct text line predictor:
                textline_predictor = Predicting(self.parameter['textline_src'], self.parameter['textline_algorithm'])

                textline_images = []
                for (n, input_file) in enumerate(self.input_files):
                    LOG.info("Predicting text lines of input file: %i / %s", n, input_file)

                    # Create a new PAGE file from the input file:
                    page_id = input_file.pageId or input_file.ID
                    pcgts = page_from_file(self.workspace.download_file(input_file))
                    page = pcgts.get_Page()

                    # Get binarized and deskewed image from PAGE:
                    page_image, page_xywh, _ = self.workspace.image_from_page(
                        page,
                        page_id,
                        feature_selector="binarized,deskewed"
                    )

                    # Convert PIL to cv2 (RGB):
                    page_image, _ = pil_to_cv2_rgb(page_image)

                    # Get labels per-pixel and map them to grayscale:
                    textline_images.append(textline_predictor.predict(page_image) * 255)

                # Close session for text lines:
                textline_predictor.session.close()
            else:
                # Get prediction images from given file group:
                textline_input_files = self.workspace.mets.find_files(
                    fileGrp=self.parameter['textline_src']
                )

                textline_images = []
                for (n, input_file) in enumerate(textline_input_files):
                    LOG.info("Retrieving cached text line prediction of input file: %i / %s", n, input_file)

                    # Create a new PAGE file from the input file:
                    page_id = input_file.pageId or input_file.ID
                    pcgts = page_from_file(self.workspace.download_file(input_file))
                    page = pcgts.get_Page()

                    # Get binarized and deskewed image from PAGE:
                    page_image, page_xywh, _ = self.workspace.image_from_page(
                        page,
                        page_id
                    )

                    # Convert PIL to cv2 (gray):
                    page_image, _ = pil_to_cv2_gray(page_image, bg_color=0)

                    textline_images.append(page_image)
        else:
            if os.path.isfile(self.parameter['textline_src']):
                # Construct text line predictor:
                textline_predictor = Predicting(self.parameter['textline_src'], self.parameter['textline_algorithm'])

                textline_images = []
                for (n, input_file) in enumerate(self.input_files):
                    LOG.info("Predicting text lines of input file: %i / %s", n, input_file)

                    # Create a new PAGE file from the input file:
                    page_id = input_file.pageId or input_file.ID
                    pcgts = page_from_file(self.workspace.download_file(input_file))
                    page = pcgts.get_Page()

                    # Get original image from PAGE:
                    page_image, page_xywh, _ = self.workspace.image_from_page(
                        page,
                        page_id,
                        feature_filter="binarized,deskewed,cropped"
                    )

                    regions = page.get_TextRegion()

                    textline_images.append([])
                    for region_idx, region in enumerate(regions):
                        # Get binarized and deskewed image from text region:
                        region_image, region_xywh = self.workspace.image_from_segment(
                            region,
                            page_image,
                            page_xywh,
                            feature_selector="binarized,deskewed"
                        )

                        # Convert PIL to cv2 (RGB):
                        region_image, _ = pil_to_cv2_rgb(region_image)

                        # Get labels per-pixel and map them to grayscale:
                        textline_images[n].append(textline_predictor.predict(region_image) * 255)

                # Close session for text lines:
                textline_predictor.session.close()
            else:
                # Get prediction images from given file group:
                textline_input_files = self.workspace.mets.find_files(
                    fileGrp=self.parameter['textline_src']
                )

                textline_images = []
                for (n, input_file) in enumerate(textline_input_files):
                    LOG.info("Retrieving cached text line prediction of input file: %i / %s", n, input_file)

                    # Create a new PAGE file from the input file:
                    page_id = input_file.pageId or input_file.ID
                    pcgts = page_from_file(self.workspace.download_file(input_file))
                    page = pcgts.get_Page()

                    # Get original image from PAGE:
                    page_image, page_xywh, _ = self.workspace.image_from_page(
                        page,
                        page_id,
                        feature_filter="binarized,deskewed,cropped"
                    )

                    regions = page.get_TextRegion()

                    textline_images.append([])
                    for region_idx, region in enumerate(regions):
                        # Get binarized and deskewed image from text region:
                        region_image, region_xywh = self.workspace.image_from_segment(
                            region,
                            page_image,
                            page_xywh,
                            feature_selector="binarized,deskewed"
                        )

                        # Convert PIL to cv2 (grayscale):
                        region_image, _ = pil_to_cv2_gray(region_image, bg_color=0)

                        textline_images[n].append(region_image)

        for (n, input_file) in enumerate(self.input_files):
            LOG.info("Processing input file: %i / %s", n, input_file)

            # Create a new PAGE file from the input file:
            page_id = input_file.pageId or input_file.ID
            pcgts = page_from_file(self.workspace.download_file(input_file))
            page = pcgts.get_Page()

            if self.parameter['operation_level'] == "page":
                # Get binarized and deskewed image from PAGE:
                page_image, page_xywh, _ = self.workspace.image_from_page(
                    page,
                    page_id,
                    feature_selector="binarized,deskewed"
                )

                self._process_page(
                    page,
                    page_image,
                    page_xywh,
                    page_id,
                    n,
                    input_file,
                    textregion_images[n],
                    textline_images[n]
                )
            elif self.parameter['operation_level'] == "region":
                # Get binarized and deskewed image from PAGE:
                page_image, page_xywh, _ = self.workspace.image_from_page(
                    page,
                    page_id,
                    feature_selector="binarized,deskewed"
                )

                # DEBUG ONLY
                # Get original image from PAGE:
                page_image_cv2, _, _ = self.workspace.image_from_page(
                    page,
                    page_id,
                    feature_filter="binarized,deskewed"
                )
                # Convert PIL to cv2 (RGB):
                page_image_cv2, alpha = pil_to_cv2_rgb(page_image_cv2)
                ############

                regions = page.get_TextRegion()

                for region_idx, region in enumerate(regions):
                    # Get binarized and deskewed image from text region:
                    region_image, region_xywh = self.workspace.image_from_segment(
                        region,
                        page_image,
                        page_xywh,
                        feature_selector="binarized,deskewed"
                    )

                    self._process_region(
                        page,
                        page_image_cv2,
                        region,
                        region_idx,
                        region_image,
                        region_xywh,
                        page_id,
                        n,
                        input_file,
                        textline_images[n][region_idx]
                    )

                # DEBUG ONLY
                # Convert cv2 to PIL (RGB):
                page_image = cv2_to_pil_rgb(page_image_cv2, alpha)

                # Get file ID of image to be saved:
                file_id = input_file.ID.replace(self.input_file_grp, self.image_grp)

                if file_id == input_file.ID:
                    file_id = concat_padded(self.image_grp, n)

                file_id += "_regions"

                # Save image:
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

            if file_id == input_file.ID:
                file_id = concat_padded(self.page_grp, n)

            # Save XML PAGE:
            self.workspace.add_file(
                 ID=file_id,
                 file_grp=self.page_grp,
                 pageId=page_id,
                 mimetype=MIMETYPE_PAGE,
                 local_filename=os.path.join(self.output_file_grp, file_id)+".xml",
                 content=to_xml(pcgts)
            )
