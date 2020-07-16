from gbn.lib.util import resolve_box, draw_box, draw_polygon, pil_to_cv2_rgb, cv2_to_pil_rgb, pil_to_cv2_gray, cv2_to_pil_gray, gray_to_bgr, binary_to_mask
from gbn.lib.predict import Predicting
from gbn.lib.extract import Extracting
from gbn.tool import OCRD_TOOL

from ocrd import Processor
from ocrd_modelfactory import page_from_file
from ocrd_models.ocrd_page import to_xml
from ocrd_models.ocrd_page_generateds import AlternativeImageType, CoordsType, LabelsType, LabelType, MetadataItemType, TextLineType, TextRegionType
from ocrd_utils import concat_padded, coordinates_for_segment, getLogger, MIMETYPE_PAGE, points_from_polygon

from os.path import realpath, join

TOOL = "ocrd-gbn-sbb-segment"
LOG = getLogger("processor.OcrdGbnSbbSegment")

FALLBACK_FILEGRP_IMG = "OCR-D-IMG-SEG"
FILEGRP_PRED = "OCR-D-IMG-PRED"

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

    def _save_segment_image(self, segment, segment_image, segment_suffix, comments, file_grp=None):
        # If no file group specified, use default group for images:
        if file_grp is None:
            file_grp = self.image_grp

        # Get file ID of image to be saved:
        file_id = self.input_file.ID.replace(self.input_file_grp, file_grp)

        if file_id == self.input_file.ID:
            file_id = concat_padded(file_grp, self.n)

        # Concatenate suffix to ID:
        file_id += segment_suffix

        # Save image:
        file_path = self.workspace.save_image_file(
            segment_image,
            file_id,
            page_id=self.page_id,
            file_grp=file_grp
        )

        # Add metadata about saved image:
        segment.add_AlternativeImage(
            AlternativeImageType(
                filename=file_path,
                comments=comments
            )
        )

    def _predict_segment(self, segment, segment_image, segment_suffix, predictor):
        # Convert PIL to cv2 (RGB):
        segment_image, alpha = pil_to_cv2_rgb(segment_image)

        # Get labels per-pixel and map them to grayscale:
        segment_prediction = predictor.predict(segment_image) * 255

        # Convert cv2 to PIL (grayscale):
        segment_prediction = cv2_to_pil_gray(segment_prediction, alpha)

        # Save prediction image as AlternativeImage of segment, setting the 'comments' field to the model path:
        self._save_segment_image(
            segment,
            segment_prediction,
            segment_suffix + predictor.model_path.translate(str.maketrans({'/': '_', '.': '_'})),
            predictor.model_path,
            file_grp=FILEGRP_PRED
        )

        return segment_prediction

    def _get_page_prediction(self, predictor, page_selector="", page_filter="", prediction_filter=""):
        # If caching is enabled:
        if self.parameter['caching']:
            try:
                # Try to cache prediction image from PAGE:
                page_prediction, page_xywh, _ = self.workspace.image_from_page(
                    self.page,
                    self.page_id,
                    feature_selector=predictor.model_path,
                    feature_filter=prediction_filter
                )

                return page_prediction, page_xywh
            except:
                LOG.info(
                    "Unable to cache predictions of input file %i / %s given model %s. Performing prediction instead",
                    self.n,
                    self.input_file,
                    predictor.model_path
                )

        # Get image from PAGE:
        page_image, page_xywh, _ = self.workspace.image_from_page(
            self.page,
            self.page_id,
            feature_selector=page_selector,
            feature_filter=page_filter
        )

        # Perform prediction:
        page_prediction = self._predict_segment(
            self.page,
            page_image,
            "",
            predictor
        )

        return page_prediction, page_xywh

    def _get_segment_prediction(self, segment, segment_suffix, predictor, segment_selector="", segment_filter="", prediction_filter=""):
        # If caching is enabled:
        if self.parameter['caching']:
            try:
                # Try to cache prediction image from PAGE:
                page_prediction, page_xywh, _ = self.workspace.image_from_page(
                    self.page,
                    self.page_id,
                    feature_selector=predictor.model_path,
                    feature_filter=prediction_filter
                )

                # Try to cache prediction image from segment:
                segment_prediction, segment_xywh, _ = self.workspace.image_from_segment(
                    segment,
                    page_prediction,
                    page_xywh,
                    feature_selector=predictor.model_path,
                    feature_filter=prediction_filter
                )

                return segment_prediction, segment_xywh
            except:
                LOG.info(
                    "Unable to cache predictions of input file %i / %s given model %s. Performing prediction instead",
                    self.n,
                    self.input_file,
                    predictor.model_path
                )

        # Get image from PAGE:
        page_image, page_xywh, _ = self.workspace.image_from_page(
            self.page,
            self.page_id,
            feature_selector=segment_selector,
            feature_filter=segment_filter
        )

        # Get image from segment:
        segment_image, segment_xywh = self.workspace.image_from_segment(
            segment,
            page_image,
            page_xywh,
            feature_selector=segment_selector,
            feature_filter=segment_filter
        )

        # Perform prediction:
        segment_prediction = self._predict_segment(
            segment,
            segment_image,
            segment_suffix,
            predictor
        )

        return segment_prediction, segment_xywh

    def _process_page(self, page, page_image, page_xywh, textregions_prediction, textlines_prediction):
        # Convert PIL to cv2 (grayscale):
        textregions_prediction, _ = pil_to_cv2_gray(textregions_prediction, bg_color=0)

        # Construct characteristic extractor for text region prediction:
        textregions_extractor = Extracting(
            textregions_prediction,
            contour_area_filter=(
                self.parameter['min_particle_size'] * page.get_imageHeight() * page.get_imageWidth(),
                self.parameter['max_particle_size'] * page.get_imageHeight() * page.get_imageWidth()
            )
        )

        # Erode and dilate prediction image to separate "weakly connected" regions:
        textregions_extractor.erode(4)
        textregions_extractor.dilate(4)

        # Analyse contours and boxes:
        textregions_extractor.analyse_contours()

        # Filter out contours inside other contours:
        textregions_extractor.filter_by_hierarchy()

        # Filter out contours that do not compose valid polygons:
        textregions_extractor.filter_by_geometry()

        # Filter contours by area:
        textregions_extractor.filter_by_area()

        # Convert PIL to cv2 (grayscale):
        textlines_prediction, _ = pil_to_cv2_gray(textlines_prediction, bg_color=0)

        # Construct characteristic extractor for text line prediction:
        textlines_extractor = Extracting(
            textlines_prediction,
            fg_density_filter=(
                self.parameter['min_textlines_density'],
                self.parameter['max_textlines_density']
            )
        )

        # Load bounding boxes of text regions into the text line extractor:
        textlines_extractor.import_boxes(textregions_extractor.export_boxes())

        # Filter boxes by textline density:
        textlines_extractor.filter_by_foreground_density()

        # Merge overlapping:
        textlines_extractor.merge_overlapping_boxes()

        # Segment clearly distinct regions of the page vertically using the pixels predicted as text lines:
        textlines_extractor.split_boxes_by_continuity(axis=1, global_projection=True)

        # Segment separate text regions horizontally using the pixels predicted as text lines:
        textlines_extractor.split_boxes_by_continuity(axis=0)
        textlines_extractor.split_boxes_by_standard_deviation(axis=0, n=4)

        # Dilate text line predictions a bit to join ones that are close to each other:
        textlines_extractor.dilate(2)

        # Segment separate text regions vertically using the pixels predicted as text lines:
        textlines_extractor.split_boxes_by_continuity(axis=1)

        # Resize text regions horizontally using the pixels predicted as text lines:
        textlines_extractor.split_boxes_by_continuity(axis=0)

        # Filter boxes by textline density:
        textlines_extractor.filter_by_foreground_density()

        # Save rectangles of boxes as text regions:
        for region_idx, region_box in enumerate(textlines_extractor.boxes):
            region_suffix = "_region%04d" % region_idx

            x0, y0, x1, y1 = resolve_box(region_box)

            polygon = [
                [x0, y0],
                [x1, y0],
                [x1, y1],
                [x0, y1]
            ]

            # Convert back to absolute (page) coordinates:
            polygon = coordinates_for_segment(polygon, page_image, page_xywh)

            # Save text region:
            page.add_TextRegion(
                TextRegionType(
                    id=self.page_id+region_suffix,
                    Coords=CoordsType(
                        points=points_from_polygon(polygon)
                    )
                )
            )

        if self.parameter['visualization']:
            # Convert PIL to cv2 (grayscale):
            page_image, alpha = pil_to_cv2_gray(page_image)

            # Get masks of both binary image (foreground prediction) and text lines prediction:
            prediction_mask = binary_to_mask(textlines_prediction)
            page_mask = binary_to_mask(page_image)

            # Convert to BGR:
            visualization = gray_to_bgr(page_image)

            # Generate visualization:
            visualization[prediction_mask == False] = (0, 0, 0) # Background: Black
            visualization[prediction_mask == True] = (255, 0, 0) # Text lines: Blue
            visualization[page_mask == False] = (255, 255, 255) # Foreground: White

            for region_box in textlines_extractor.boxes:
                draw_box(visualization, region_box, (0, 255, 0), 3) # Text region rectangles: Green

            # Convert cv2 to PIL (RGB):
            visualization = cv2_to_pil_rgb(visualization, alpha)

            # Save visualization as AlternativeImage:
            self._save_segment_image(
                page,
                visualization,
                "_page_level",
                "visualization"
            )

    def _process_region(self, page, page_image, region, region_image, region_xywh, region_suffix, textlines_prediction):
        # Convert PIL to cv2 (grayscale):
        textlines_prediction, _ = pil_to_cv2_gray(textlines_prediction, bg_color=0)

        # Construct characteristic extractor for text line prediction:
        textlines_extractor = Extracting(
            textlines_prediction,
            fg_density_filter=(
                self.parameter['min_textlines_density'],
                self.parameter['max_textlines_density']
            )
        )

        # TODO: Achieve this through API
        # Filter whole region by textline density:
        density = textlines_extractor.get_foreground_density()
        if density < self.parameter['min_textlines_density'] or self.parameter['max_textlines_density'] < density:
            return

        # Split region into line frames:
        frames = textlines_extractor.split_image_into_frames(axis=1)

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
                    self.parameter['min_textlines_density'],
                    self.parameter['max_textlines_density']
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

            line_id = self.page_id + region_suffix + "_line%04d" % idx

            # Save text line:
            region.add_TextLine(
                TextLineType(
                    id=line_id,
                    Coords=CoordsType(
                        points=points_from_polygon(polygon)
                    )
                )
            )

            if self.parameter['visualization']:
                # Convert PIL to cv2:
                visualization, alpha = pil_to_cv2_rgb(page_image)

                # Generate visualization
                draw_polygon(visualization, polygon, (0, 127, 0), 2) # Green

                # Convert cv2 to PIL (RGB):
                visualization = cv2_to_pil_rgb(visualization, alpha)

        return visualization

    def process(self):
        if self.parameter['textregions_model'] is None:
            if self.parameter['operation_level'] == "page":
                LOG.error("Operation level 'page' requires a path to the text regions prediction model")
                quit()
        else:
            # Ensure path to model is absolute:
            self.parameter['textregions_model'] = realpath(self.parameter['textregions_model'])

            # Construct predictor object:
            textregions_predictor = Predicting(self.parameter['textregions_model'], self.parameter['textlines_shaping'])

        # Ensure path to model is absolute:
        self.parameter['textlines_model'] = realpath(self.parameter['textlines_model'])

        # Construct predictor object:
        textlines_predictor = Predicting(self.parameter['textlines_model'], self.parameter['textlines_shaping'])

        for (self.n, self.input_file) in enumerate(self.input_files):
            LOG.info("Processing input file: %i / %s", self.n, self.input_file)

            # Create a new PAGE file from the input file:
            self.page_id = self.input_file.pageId or self.input_file.ID
            self.pcgts = page_from_file(self.workspace.download_file(self.input_file))
            self.page = self.pcgts.get_Page()

            # Get binarized and deskewed image from PAGE:
            page_image, page_xywh, _ = self.workspace.image_from_page(
                self.page,
                self.page_id,
                feature_selector="binarized,deskewed"
            )

            if self.parameter['operation_level'] == "page":
                # Get TextRegion prediction image:
                textregions_prediction, _ = self._get_page_prediction(
                    textregions_predictor,
                    page_selector="binarized,deskewed"
                )

                # Get TextLine prediction image:
                textlines_prediction, _ = self._get_page_prediction(
                    textlines_predictor,
                    page_selector="binarized,deskewed"
                )

                # Segment page:
                self._process_page(
                    self.page,
                    page_image,
                    page_xywh,
                    textregions_prediction,
                    textlines_prediction
                )
            elif self.parameter['operation_level'] == "region":
                # TODO: Place visualization stuff inside a _process method
                # Get original image from PAGE:
                visualization, _, _ = self.workspace.image_from_page(
                    self.page,
                    self.page_id,
                    feature_filter="binarized,deskewed"
                )

                regions = self.page.get_TextRegion()

                for region_idx, region in enumerate(regions):
                    region_suffix = "_region%04d" % region_idx

                    # Get TextLine prediction image:
                    textlines_prediction, _ = self._get_segment_prediction(
                        region,
                        region_suffix,
                        textlines_predictor,
                        segment_selector="binarized,deskewed"
                    )

                    # Get binarized and deskewed image from TextRegion:
                    region_image, region_xywh, _ = self.workspace.image_from_segment(
                        region,
                        page_image,
                        page_xywh,
                        feature_selector="binarized,deskewed"
                    )

                    # Segment text regions:
                    visualization = self._process_region(
                        self.page,
                        visualization,
                        region,
                        region_image,
                        region_xywh,
                        region_suffix,
                        textlines_prediction
                    )

                if self.parameter['visualization']:
                    # Save visualization as AlternativeImage:
                    self._save_segment_image(
                        self.page,
                        visualization,
                        "_region_level",
                        "visualization"
                    )

            # Add metadata about this operation:
            metadata = self.pcgts.get_Metadata()
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
            file_id = self.input_file.ID.replace(self.input_file_grp, self.page_grp)

            if file_id == self.input_file.ID:
                file_id = concat_padded(self.page_grp, self.n)

            # Save XML PAGE:
            self.workspace.add_file(
                 ID=file_id,
                 file_grp=self.page_grp,
                 pageId=self.page_id,
                 mimetype=MIMETYPE_PAGE,
                 local_filename=join(self.output_file_grp, file_id)+".xml",
                 content=to_xml(self.pcgts)
            )
