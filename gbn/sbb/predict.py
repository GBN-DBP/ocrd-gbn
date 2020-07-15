from gbn.lib.util import pil_to_cv2_rgb, cv2_to_pil_gray
from gbn.lib.predict import Predicting
from gbn.tool import OCRD_TOOL

from ocrd import Processor
from ocrd_modelfactory import page_from_file
from ocrd_models.ocrd_page import to_xml
from ocrd_models.ocrd_page_generateds import AlternativeImageType, LabelsType, LabelType, MetadataItemType
from ocrd_utils import concat_padded, getLogger, MIMETYPE_PAGE

from os.path import realpath, join

TOOL = "ocrd-gbn-sbb-predict"
LOG = getLogger("processor.OcrdGbnSbbPredict")

FILEGRP_PRED = "OCR-D-IMG-PRED"

class OcrdGbnSbbPredict(Processor):
    def __init__(self, *args, **kwargs):
        kwargs['ocrd_tool'] = OCRD_TOOL['tools'][TOOL]
        kwargs['version'] = OCRD_TOOL['version']
        super(OcrdGbnSbbPredict, self).__init__(*args, **kwargs)

        if hasattr(self, "output_file_grp"):
            try:
                self.page_grp, self.image_grp = self.output_file_grp.split(',')
                self.output_file_grp = self.page_grp
                FILEGRP_PRED = self.image_grp
            except ValueError:
                self.page_grp = self.output_file_grp
                self.image_grp = FILEGRP_PRED
                LOG.info("No output file group for predictions specified, falling back to '%s'", FILEGRP_PRED)

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
                # Get image from PAGE:
                page_image, _, _ = self.workspace.image_from_page(
                    self.page,
                    self.page_id
                )

                # Perform prediction:
                self._predict_segment(
                    self.page,
                    page_image,
                    "",
                    predictor
                )
            elif self.parameter['operation_level'] == "region":
                # Get image from PAGE:
                page_image, page_xywh, _ = self.workspace.image_from_page(
                    self.page,
                    self.page_id
                )

                regions = self.page.get_TextRegion()

                for region_idx, region in enumerate(regions):
                    region_suffix = "_region%04d" % region_idx

                    # Get image from TextRegion:
                    region_image, _ = self.workspace.image_from_segment(
                        region,
                        page_image,
                        page_xywh
                    )

                    # Perform prediction:
                    self._predict_segment(
                        region,
                        region_image,
                        region_suffix,
                        predictor
                    )
            elif self.parameter['operation_level'] == "line":
                # Get image from PAGE:
                page_image, page_xywh, _ = self.workspace.image_from_page(
                    self.page,
                    self.page_id
                )

                regions = self.page.get_TextRegion()

                for region_idx, region in enumerate(regions):
                    region_suffix = "_region%04d" % region_idx

                    lines = region.get_TextLine()

                    for line_idx, line in enumerate(lines):
                        line_suffix = region_suffix + ("_line%04d" % line_idx)

                        # Get image from TextLine:
                        line_image, _ = self.workspace.image_from_segment(
                            line,
                            page_image,
                            page_xywh
                        )

                        # Perform prediction:
                        self._predict_segment(
                            line,
                            line_image,
                            line_suffix,
                            predictor
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
