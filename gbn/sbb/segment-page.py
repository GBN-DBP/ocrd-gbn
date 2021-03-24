import os.path

import ocrd_modelfactory
import ocrd_models.ocrd_page as ocrd_page
import ocrd_models.ocrd_page_generateds as ocrd_page_gends
import ocrd_utils

from gbn.processor import OcrdGbnProcessor
from gbn.lib.dl import Model
from gbn.lib.struct import Contour, Polygon
from gbn.lib.util import pil_to_cv2_rgb


class OcrdGbnSbbSegmentPage(OcrdGbnProcessor):
    tool = "ocrd-gbn-sbb-segment-page"
    log = ocrd_utils.getLogger("processor.OcrdGbnSbbSegmentPage")

    fallback_image_filegrp = "OCR-D-IMG-SEG-PAGE"

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

            # Get Region prediction for page:
            region_prediction = model.predict(page_image_cv2)

            # Get Border from PAGE:
            border = page.get_Border()

            # Get PrintSpace from PAGE:
            print_space = page.get_PrintSpace()

            if print_space is not None:
                # Get PrintSpace polygon:
                print_space_polygon = Polygon(
                    print_space.get_Coords().get_points()
                )

                # Get Region prediction inside the PrintSpace:
                region_prediction = region_prediction.crop(print_space_polygon)

            elif border is not None:
                # Get Border polygon:
                border_polygon = Polygon(border.get_Coords().get_points())

                # Get Region prediction inside the Border:
                region_prediction = region_prediction.crop(border_polygon)

            for clss, label in self.parameter['classes']:
                # Find contours of given class in prediction:
                region_contours = Contour.from_image(
                    region_prediction.img,
                    label
                )

                # Filter out child contours:
                region_contours = list(
                    filter(
                        lambda cnt: not cnt.is_child(),
                        region_contours
                    )
                )

                # Filter out invalid polygons:
                region_contours = list(
                    filter(
                        lambda cnt: cnt.polygon.is_valid(),
                        region_contours
                    )
                )

                # Add metadata about regions:
                for region_idx, region_cnt in enumerate(region_contours):
                    region_id = "_" + clss + "%04d" % region_idx

                    self._add_Region(
                        page,
                        page_image,
                        page_xywh,
                        page_id,
                        clss,
                        region_cnt.polygon.points,
                        region_id
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
