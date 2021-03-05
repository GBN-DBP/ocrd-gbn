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
        # Ensure path to model is absolute:
        self.parameter['model'] = realpath(self.parameter['model'])

        # Construct Model object for prediction:
        model = Model(self.parameter['model'], self.parameter['shaping'])

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

            # Get Region prediction for page:
            region_prediction = model.predict(page_image_cv2)

            # Get Border from PAGE:
            border = page.get_Border()

            # Get PrintSpace from PAGE:
            print_space = page.get_PrintSpace()

            if print_space is not None:
                # Get PrintSpace polygon:
                print_space_polygon = Polygon(print_space.get_Coords().get_points())

                # Get Region prediction inside the PrintSpace:
                region_prediction = region_prediction.crop(print_space_polygon)

            elif border is not None:
                # Get Border polygon:
                border_polygon = Polygon(border.get_Coords().get_points())

                # Get Region prediction inside the Border:
                region_prediction = region_prediction.crop(border_polygon)

            # Find TextRegion contours of prediction:
            text_region_contours = Contour.from_image(region_prediction.img, 1)

            # Filter out child contours:
            text_region_contours = list(filter(lambda cnt: not cnt.is_child(), text_region_contours))

            # Filter out invalid polygons:
            text_region_contours = list(filter(lambda cnt: cnt.polygon.is_valid(), text_region_contours))

            # Add metadata about TextRegions:
            for region_idx, region_cnt in enumerate(text_region_contours):
                region_id = "_text_region%04d" % region_idx

                self._add_TextRegion(
                    page,
                    page_image,
                    page_xywh,
                    page_id,
                    region_cnt.polygon.points,
                    region_id
                )

            # Find ImageRegion contours of prediction:
            image_region_contours = Contour.from_image(region_prediction.img, 2)

            # Filter out child contours:
            image_region_contours = list(filter(lambda cnt: not cnt.is_child(), image_region_contours))

            # Filter out invalid polygons:
            image_region_contours = list(filter(lambda cnt: cnt.polygon.is_valid(), image_region_contours))

            # Add metadata about ImageRegions:
            for region_idx, region_cnt in enumerate(image_region_contours):
                region_id = "_image_region%04d" % region_idx

                self._add_ImageRegion(
                    page,
                    page_image,
                    page_xywh,
                    page_id,
                    region_cnt.polygon.points,
                    region_id
                )

            # Find GraphicRegion contours of prediction:
            graphic_region_contours = Contour.from_image(region_prediction.img, 3)

            # Filter out child contours:
            graphic_region_contours = list(filter(lambda cnt: not cnt.is_child(), graphic_region_contours))

            # Filter out invalid polygons:
            graphic_region_contours = list(filter(lambda cnt: cnt.polygon.is_valid(), graphic_region_contours))

            # Add metadata about GraphicRegions:
            for region_idx, region_cnt in enumerate(graphic_region_contours):
                region_id = "_graphic_region%04d" % region_idx

                self._add_GraphicRegion(
                    page,
                    page_image,
                    page_xywh,
                    page_id,
                    region_cnt.polygon.points,
                    region_id
                )

            # Find SeparatorRegion contours of prediction:
            separator_region_contours = Contour.from_image(region_prediction.img, 4)

            # Filter out child contours:
            separator_region_contours = list(filter(lambda cnt: not cnt.is_child(), separator_region_contours))

            # Filter out invalid polygons:
            separator_region_contours = list(filter(lambda cnt: cnt.polygon.is_valid(), separator_region_contours))

            # Add metadata about SeparatorRegions:
            for region_idx, region_cnt in enumerate(separator_region_contours):
                region_id = "_separator_region%04d" % region_idx

                self._add_SeparatorRegion(
                    page,
                    page_image,
                    page_xywh,
                    page_id,
                    region_cnt.polygon.points,
                    region_id
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
