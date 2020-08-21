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

class OcrdGbnSbbSegment(OcrdGbnSbbPredict):
    tool = "ocrd-gbn-sbb-segment"
    log = getLogger("processor.OcrdGbnSbbSegment")

    fallback_image_filegrp = "OCR-D-IMG-SEG"

    def process(self):
        # Ensure path to TextRegion model is absolute:
        self.parameter['region_model'] = realpath(self.parameter['region_model'])

        # Construct Model object for TextRegion prediction:
        region_model = Model(self.parameter['region_model'], self.parameter['region_shaping'])

        # Ensure path to TextLine model is absolute:
        self.parameter['line_model'] = realpath(self.parameter['line_model'])

        # Construct Model object for TextLine prediction:
        line_model = Model(self.parameter['line_model'], self.parameter['line_shaping'])

        for (self.page_num, self.input_file) in enumerate(self.input_files):
            self.log.info("Processing input file: %i / %s", self.page_num, self.input_file)

            # Create a new PAGE file from the input file:
            page_id = self.input_file.pageId or self.input_file.ID
            pcgts = page_from_file(self.workspace.download_file(self.input_file))
            page = pcgts.get_Page()

            # Get image from PAGE:
            page_image, page_xywh, _ = self.workspace.image_from_page(
                page,
                page_id
            )

            # Convert PIL to cv2 (RGB):
            page_image_cv2, _ = pil_to_cv2_rgb(page_image)

            # Get TextRegion prediction for page:
            region_prediction = region_model.predict(page_image_cv2)

            # Find contours of prediction:
            region_contours = Contour.from_image(region_prediction.img)

            # Filter out child contours:
            region_contours = list(filter(lambda cnt: not cnt.is_child(), region_contours))

            # Filter out invalid polygons:
            region_contours = list(filter(lambda cnt: cnt.polygon.is_valid(), region_contours))

            # Get TextLine prediction for page:
            line_prediction_page = line_model.predict(page_image_cv2)

            # Add metadata about TextRegions:
            for region_idx, region_cnt in enumerate(region_contours):
                region_id = "_region%04d" % region_idx

                self._add_TextRegion(
                    page,
                    page_image,
                    page_xywh,
                    page_id,
                    region_cnt.polygon.points,
                    region_id
                )

            # Retrieve added TextRegions:
            regions = page.get_TextRegion()

            for region_idx, (region_cnt, region) in enumerate(zip(region_contours, regions)):
                region_id = "_region%04d" % region_idx

                # Get image from TextRegion:
                region_image, region_xywh = self.workspace.image_from_segment(
                    region,
                    page_image,
                    page_xywh
                )

                # Get TextLine prediction for TextRegion:
                line_prediction_region = line_prediction_page.crop(region_cnt.polygon)

                # Find contours of prediction:
                line_contours = Contour.from_image(line_prediction_region.img)

                # Filter out child contours:
                line_contours = list(filter(lambda cnt: not cnt.is_child(), line_contours))

                # Filter out invalid polygons:
                line_contours = list(filter(lambda cnt: cnt.polygon.is_valid(), line_contours))

                # Add metadata about TextLines:
                for line_idx, line_cnt in enumerate(line_contours):
                    line_id = "_line%04d" % line_idx

                    self._add_TextLine(
                        page_id,
                        region,
                        region_image,
                        region_xywh,
                        region_id,
                        line_cnt.polygon.points,
                        line_id
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
