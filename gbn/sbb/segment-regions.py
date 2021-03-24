import os.path

import ocrd_modelfactory
import ocrd_models.ocrd_page as ocrd_page
import ocrd_models.ocrd_page_generateds as ocrd_page_gends
import ocrd_utils

from gbn.processor import OcrdGbnProcessor
from gbn.lib.dl import Model
from gbn.lib.struct import Contour, Polygon
from gbn.lib.util import pil_to_cv2_rgb


class OcrdGbnSbbSegmentRegions(OcrdGbnProcessor):
    tool = "ocrd-gbn-sbb-segment-regions"
    log = ocrd_utils.getLogger("processor.OcrdGbnSbbSegmentRegions")

    fallback_image_filegrp = "OCR-D-IMG-SEG-REGIONS"

    def process(self):
        # Ensure path to model is absolute:
        self.parameter['model'] = os.path.realpath(self.parameter['model'])

        # Construct Model object for prediction:
        model = Model(self.parameter['model'], self.parameter['shaping'])

        for (self.page_num, self.input_file) in enumerate(self.input_files):
            self.log.info(
                "Processing input file: %i / %s",
                self.page_num,
                self.input_file
            )

            # Create a new PAGE file from the input file:
            page_id = self.input_file.pageId or self.input_file.ID
            pcgts = ocrd_modelfactory.page_from_file(
                self.workspace.download_file(self.input_file)
            )
            page = pcgts.get_Page()

            # Get image from PAGE:
            page_image, page_xywh, _ = self.workspace.image_from_page(
                page,
                page_id
            )

            # Convert PIL to cv2 (RGB):
            page_image_cv2, _ = pil_to_cv2_rgb(page_image)

            # Get TextLine prediction for page:
            line_prediction = model.predict(page_image_cv2)

            regions = page.get_TextRegion()

            for region_idx, region in enumerate(regions):
                region_image, region_xywh = self.workspace.image_from_segment(
                    region,
                    page_image,
                    page_xywh
                )

                region_id = region.get_regionRef()

                # Get TextRegion polygon:
                region_polygon = Polygon(region.get_Coords().get_points())

                # Get TextLine prediction inside TextRegion:
                line_subprediction = line_prediction.crop(region_polygon)

                # Find contours of given class in prediction:
                line_contours = Contour.from_image(
                    line_subprediction.img,
                    self.parameter['classes']['TextLine']
                )

                # Filter out child contours:
                line_contours = list(
                    filter(
                        lambda cnt: not cnt.is_child(),
                        line_contours
                    )
                )

                # Filter out invalid polygons:
                line_contours = list(
                    filter(
                        lambda cnt: cnt.polygon.is_valid(),
                        line_contours
                    )
                )

                for line_idx, line_cnt in enumerate(line_contours):
                    line_id = "_TextLine" + "%04d" % line_idx

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
                ocrd_page_gends.MetadataItemType(
                    type_="processingStep",
                    name=self.ocrd_tool['steps'][0],
                    value=self.tool,
                    Labels=[
                        ocrd_page_gends.LabelsType(
                            externalModel="ocrd-tool",
                            externalId="parameters",
                            Label=[
                                ocrd_page_gends.LabelType(
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
                mimetype=ocrd_utils.MIMETYPE_PAGE,
                local_filename=os.path.join(
                    self.output_file_grp,
                    self.page_file_id
                ) + ".xml",
                content=ocrd_page.to_xml(pcgts)
            )
