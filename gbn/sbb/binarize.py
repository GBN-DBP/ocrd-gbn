import os.path

import ocrd_modelfactory
import ocrd_models.ocrd_page as ocrd_page
import ocrd_models.ocrd_page_generateds as ocrd_page_gends
import ocrd_utils

from gbn.processor import OcrdGbnProcessor
from gbn.lib.dl import Model
from gbn.lib.util import pil_to_cv2_rgb, cv2_to_pil_gray


class OcrdGbnSbbBinarize(OcrdGbnProcessor):
    tool = "ocrd-gbn-sbb-binarize"
    log = ocrd_utils.getLogger("processor.OcrdGbnSbbBinarize")

    fallback_image_filegrp = "OCR-D-IMG-BIN"

    def process(self):
        # Ensure path to model is absolute:
        self.parameter['model'] = os.path.realpath(self.parameter['model'])

        # Construct Model object:
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

            if self.parameter['operation-level'] == "page":
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
                page_prediction = cv2_to_pil_gray(
                    page_prediction.to_binary_image(),
                    alpha=alpha
                )

                self._add_AlternativeImage(
                    page_id,
                    page,
                    page_prediction,
                    page_xywh,
                    "",
                    "binarized"
                )

            elif self.parameter['operation-level'] == "regions":
                # Get image from PAGE:
                page_image, page_xywh, _ = self.workspace.image_from_page(
                    page,
                    page_id
                )

                regions = page.get_TextRegion()

                for region in regions:
                    region_id = region.get_id()

                    # Get image from TextRegion:
                    region_image, region_xywh = \
                        self.workspace.image_from_segment(
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
                    region_prediction = cv2_to_pil_gray(
                        region_prediction.to_binary_image(), alpha=alpha)

                    self._add_AlternativeImage(
                        page_id,
                        region,
                        region_prediction,
                        region_xywh,
                        region_id,
                        "binarized"
                    )

            elif self.parameter['operation-level'] == "lines":
                # Get image from PAGE:
                page_image, page_xywh, _ = self.workspace.image_from_page(
                    page,
                    page_id
                )

                regions = page.get_TextRegion()

                for region in regions:
                    region_id = region.get_id()

                    lines = region.get_TextLine()

                    for line_idx, line in enumerate(lines):
                        line_id = "_region%04d" % line_idx

                        # Get image from TextLine:
                        line_image, line_xywh = \
                            self.workspace.image_from_segment(
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
                        line_prediction = cv2_to_pil_gray(
                            line_prediction.to_binary_image(),
                            alpha=alpha
                        )

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

            # Save PAGE-XML file:
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
