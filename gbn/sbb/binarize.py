from gbn.lib.dl import Model, Prediction
from gbn.lib.struct import Contour, Polygon
from gbn.lib.util import pil_to_cv2_rgb, cv2_to_pil_gray
from gbn.tool import OCRD_TOOL
from gbn.sbb.predict import OcrdGbnSbbPredict

from ocrd_modelfactory import page_from_file
from ocrd_models.ocrd_page import to_xml
from ocrd_models.ocrd_page_generateds import AlternativeImageType, BorderType, CoordsType, LabelsType, LabelType, MetadataItemType, TextLineType, TextRegionType
from ocrd_utils import concat_padded, coordinates_for_segment, getLogger, MIMETYPE_PAGE, points_from_polygon

from os.path import realpath, join

class OcrdGbnSbbBinarize(OcrdGbnSbbPredict):
    tool = "ocrd-gbn-sbb-binarize"
    log = getLogger("processor.OcrdGbnSbbBinarize")

    fallback_image_filegrp = "OCR-D-IMG-BIN"

    def process(self):
        # Ensure path to model is absolute:
        self.parameter['model'] = realpath(self.parameter['model'])

        # Construct Model object:
        model = Model(self.parameter['model'], self.parameter['shaping'])

        for (self.page_num, self.input_file) in enumerate(self.input_files):
            self.log.info("Processing input file: %i / %s", self.page_num, self.input_file)

            # Create a new PAGE file from the input file:
            page_id = self.input_file.pageId or self.input_file.ID
            pcgts = page_from_file(self.workspace.download_file(self.input_file))
            page = pcgts.get_Page()

            if self.parameter['operation_level'] == "page":
                # Get image from PAGE:
                page_image, page_xywh, _ = self.workspace.image_from_page(
                    page,
                    page_id,
                    feature_filter="binarized"
                )

                # Convert PIL to cv2 (RGB):
                page_image, alpha = pil_to_cv2_rgb(page_image)

                # Get prediction for segment:
                page_prediction = model.predict(page_image)

                # Convert to cv2 binary image then to PIL:
                page_prediction = cv2_to_pil_gray(page_prediction.to_binary_image(), alpha=alpha)

                self._add_AlternativeImage(
                    page_id,
                    page,
                    page_prediction,
                    page_xywh,
                    "",
                    "binarized"
                )

            elif self.parameter['operation_level'] == "region":
                # Get image from PAGE:
                page_image, page_xywh, _ = self.workspace.image_from_page(
                    page,
                    page_id
                )

                regions = page.get_TextRegion()

                for region_idx, region in enumerate(regions):
                    region_id = "_region%04d" % region_idx

                    # Get image from TextRegion:
                    region_image, region_xywh = self.workspace.image_from_segment(
                        region,
                        page_image,
                        page_xywh,
                        feature_filter="binarized"
                    )

                    # Convert PIL to cv2 (RGB):
                    region_image, alpha = pil_to_cv2_rgb(region_image)

                    # Get prediction for segment:
                    region_prediction = model.predict(region_image)

                    # Convert to cv2 binary image then to PIL:
                    region_prediction = cv2_to_pil_gray(region_prediction.to_binary_image(), alpha=alpha)

                    self._add_AlternativeImage(
                        page_id,
                        region,
                        region_prediction,
                        region_xywh,
                        region_id,
                        "binarized"
                    )

            elif self.parameter['operation_level'] == "line":
                # Get image from PAGE:
                page_image, page_xywh, _ = self.workspace.image_from_page(
                    page,
                    page_id
                )

                regions = page.get_TextRegion()

                for region_idx, region in enumerate(regions):
                    region_id = "_region%04d" % region_idx

                    lines = region.get_TextLine()

                    for line_idx, line in enumerate(lines):
                        line_id = "_region%04d" % line_idx

                        # Get image from TextLine:
                        line_image, line_xywh = self.workspace.image_from_segment(
                            line,
                            page_image,
                            page_xywh,
                            feature_filter="binarized"
                        )

                        # Convert PIL to cv2 (RGB):
                        line_image, alpha = pil_to_cv2_rgb(line_image)

                        # Get prediction for segment:
                        line_prediction = model.predict(line_image)

                        # Convert to cv2 binary image then to PIL:
                        line_prediction = cv2_to_pil_gray(line_prediction.to_binary_image(), alpha=alpha)

                        self._add_AlternativeImage(
                            page_id,
                            line,
                            line_prediction,
                            line_xywh,
                            region_id+line_id,
                            "binarized"
                        )

            # Add metadata about this operation:
            metadata = pcgts.get_Metadata()
            metadata.add_MetadataItem(
                MetadataItemType(
                    type_="processingStep",
                    name=self.ocrd_tool['steps'][0],
                    value=self.tool,
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

            # Save XML PAGE:
            self.workspace.add_file(
                 ID=self.page_file_id,
                 file_grp=self.page_grp,
                 pageId=page_id,
                 mimetype=MIMETYPE_PAGE,
                 local_filename=join(self.output_file_grp, self.page_file_id)+".xml",
                 content=to_xml(pcgts)
            )
