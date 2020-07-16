from gbn.lib.util import invert_image, pil_to_cv2_rgb, pil_to_cv2_gray, cv2_to_pil_gray
from gbn.lib.predict import Predicting
from gbn.tool import OCRD_TOOL

from ocrd import Processor
from ocrd_modelfactory import page_from_file
from ocrd_models.ocrd_page import to_xml
from ocrd_models.ocrd_page_generateds import AlternativeImageType, LabelsType, LabelType, MetadataItemType
from ocrd_utils import concat_padded, getLogger, MIMETYPE_PAGE

from os.path import realpath, join

TOOL = "ocrd-gbn-sbb-binarize"
LOG = getLogger("processor.OcrdGbnSbbBinarize")

FALLBACK_FILEGRP_IMG = "OCR-D-IMG-BIN"
FILEGRP_PRED = "OCR-D-IMG-PRED"

class OcrdGbnSbbBinarize(Processor):
    def __init__(self, *args, **kwargs):
        kwargs['ocrd_tool'] = OCRD_TOOL['tools'][TOOL]
        kwargs['version'] = OCRD_TOOL['version']
        super(OcrdGbnSbbBinarize, self).__init__(*args, **kwargs)

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
                segment_prediction, segment_xywh = self.workspace.image_from_segment(
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

    def _process_segment(self, segment, segment_xywh, segment_suffix, segment_prediction):
        # Convert PIL to cv2 (grayscale):
        segment_prediction, alpha = pil_to_cv2_gray(segment_prediction, bg_color=0)

        # Invert prediction image so foreground is black:
        segment_image = invert_image(segment_prediction)

        # Convert cv2 to PIL (grayscale):
        segment_image = cv2_to_pil_gray(segment_image, alpha)

        # Save binarized image as AlternativeImage of segment:
        self._save_segment_image(
            segment,
            segment_image,
            segment_suffix,
            "binarized" if not segment_xywh['features'] else segment_xywh['features'] + ",binarized"
        )

    def process(self):
        # Ensure path to model is absolute:
        self.parameter['model'] = realpath(self.parameter['model'])

        # Construct predictor object:
        predictor = Predicting(self.parameter['model'], self.parameter['shaping'])

        for (self.n, self.input_file) in enumerate(self.input_files):
            LOG.info("Processing input file: %i / %s", self.n, self.input_file)

            # Create a new PAGE file from the input file:
            self.page_id = self.input_file.pageId or self.input_file.ID
            self.pcgts = page_from_file(self.workspace.download_file(self.input_file))
            self.page = self.pcgts.get_Page()

            if self.parameter['operation_level'] == "page":
                # Get prediction image:
                page_prediction, page_xywh = self._get_page_prediction(
                    predictor,
                    page_filter="binarized"
                )

                # Binarize page:
                self._process_segment(
                    self.page,
                    page_xywh,
                    "",
                    page_prediction
                )
            elif self.parameter['operation_level'] == "region":
                regions = self.page.get_TextRegion()

                for region_idx, region in enumerate(regions):
                    region_suffix = "_region%04d" % region_idx

                    # Get prediction image:
                    region_prediction, region_xywh = self._get_segment_prediction(
                        region,
                        region_suffix,
                        predictor,
                        segment_filter="binarized"
                    )

                    # Binarize text region:
                    self._process_segment(
                        region,
                        region_xywh,
                        region_suffix,
                        region_prediction
                    )
            elif self.parameter['operation_level'] == "line":
                regions = self.page.get_TextRegion()

                for region_idx, region in enumerate(regions):
                    region_suffix = "_region%04d" % region_idx

                    lines = region.get_TextLine()

                    for line_idx, line in enumerate(lines):
                        line_suffix = region_suffix + ("_line%04d" % line_idx)

                        # Get prediction image:
                        line_prediction, line_xywh = self._get_segment_prediction(
                            line,
                            line_suffix,
                            predictor,
                            segment_filter="binarized"
                        )

                        # Binarize text line:
                        self._process_segment(
                            line,
                            line_xywh,
                            line_suffix,
                            line_prediction
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
