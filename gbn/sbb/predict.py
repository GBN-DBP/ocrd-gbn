from gbn.lib.dl import Model, Prediction
from gbn.lib.struct import Contour, Polygon
from gbn.lib.util import pil_to_cv2_rgb, cv2_to_pil_gray
from gbn.tool import OCRD_TOOL

from ocrd import Processor
from ocrd_modelfactory import page_from_file
from ocrd_models.ocrd_page import to_xml
from ocrd_models.ocrd_page_generateds import AlternativeImageType, BorderType, CoordsType, LabelsType, LabelType, MetadataItemType, TextLineType, TextRegionType
from ocrd_utils import concat_padded, coordinates_for_segment, getLogger, MIMETYPE_PAGE, points_from_polygon

from os.path import realpath, join

class OcrdGbnSbbPredict(Processor):
    tool = "ocrd-gbn-sbb-predict"
    log = getLogger("processor.OcrdGbnSbbPredict")

    fallback_image_filegrp = "OCR-D-IMG-PRED"

    def __init__(self, *args, **kwargs):
        kwargs['ocrd_tool'] = OCRD_TOOL['tools'][self.tool]
        kwargs['version'] = OCRD_TOOL['version']
        super(OcrdGbnSbbPredict, self).__init__(*args, **kwargs)

        if hasattr(self, "output_file_grp"):
            try:
                # If image file group specified:
                self.page_grp, self.image_grp = self.output_file_grp.split(',')
                self.output_file_grp = self.page_grp
            except ValueError:
                # If image file group not specified:
                self.page_grp = self.output_file_grp
                self.image_grp = self.fallback_image_filegrp
                self.log.info(
                    "No output file group for images specified, falling back to '%s'",
                    self.fallback_image_filegrp
                )

    def file_id(self, file_grp):
        file_id = self.input_file.ID.replace(self.input_file_grp, file_grp)

        if file_id == self.input_file.ID:
            file_id = concat_padded(file_grp, self.page_num)

        return file_id

    @property
    def page_file_id(self):
        return self.file_id(self.page_grp)

    @property
    def image_file_id(self):
        return self.file_id(self.image_grp)

    def _add_AlternativeImage(self, page_id, segment, segment_image, segment_xywh, segment_id, comments):
        # Save image:
        file_path = self.workspace.save_image_file(
            segment_image,
            self.image_file_id+segment_id,
            page_id=page_id,
            file_grp=self.image_grp
        )

        # Add metadata about saved image:
        segment.add_AlternativeImage(
            AlternativeImageType(
                filename=file_path,
                comments=comments if not segment_xywh['features'] else segment_xywh['features'] + "," + comments
            )
        )

    def _set_Border(self, page, page_image, page_xywh, border_polygon):
        # Convert to absolute (page) coordinates:
        border_polygon = coordinates_for_segment(border_polygon, page_image, page_xywh)

        # Save border:
        page.set_Border(
            BorderType(
                Coords=CoordsType(
                    points=points_from_polygon(border_polygon)
                )
            )
        )

    def _add_TextRegion(self, page, page_image, page_xywh, page_id, region_polygon, region_id):
        # Convert to absolute (page) coordinates:
        region_polygon = coordinates_for_segment(region_polygon, page_image, page_xywh)

        # Save text region:
        page.add_TextRegion(
            TextRegionType(
                id=page_id+region_id,
                Coords=CoordsType(
                    points=points_from_polygon(region_polygon)
                )
            )
        )

    def _add_TextLine(self, page_id, region, region_image, region_xywh, region_id, line_polygon, line_id):
        # Convert to absolute (page) coordinates:
        line_polygon = coordinates_for_segment(line_polygon, region_image, region_xywh)

        # Save text line:
        region.add_TextLine(
            TextLineType(
                id=page_id+region_id+line_id,
                Coords=CoordsType(
                    points=points_from_polygon(line_polygon)
                )
            )
        )

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
                    page_id
                )

                # Convert PIL to cv2 (RGB):
                page_image, alpha = pil_to_cv2_rgb(page_image)

                # Get prediction for segment:
                page_prediction = model.predict(page_image)

                if self.parameter['type'] == "AlternativeImageType":
                    # Convert to cv2 binary image then to PIL:
                    page_prediction = cv2_to_pil_gray(page_prediction.to_binary_image(), alpha=alpha)

                    self._add_AlternativeImage(page_id, page, page_image, page_xywh, "", "")

                elif self.parameter['type'] == "BorderType":
                    # Find contours of prediction:
                    contours = Contour.from_image(page_prediction.img)

                    # Filter out child contours:
                    contours = list(filter(lambda cnt: not cnt.is_child(), contours))

                    # Filter out invalid polygons:
                    contours = list(filter(lambda cnt: cnt.polygon.is_valid(), contours))

                    # Sort contours by area:
                    contours = sorted(contours, key=lambda cnt: cnt.area)

                    # Get polygon of largest contour:
                    border_polygon = contours[-1].polygon

                    self._set_Border(page, page_image, page_xywh, border_polygon)

                elif self.parameter['type'] == "TextRegionType":
                    # Find contours of prediction:
                    contours = Contour.from_image(page_prediction.img)

                    # Filter out child contours:
                    contours = list(filter(lambda cnt: not cnt.is_child(), contours))

                    # Filter out invalid polygons:
                    contours = list(filter(lambda cnt: cnt.polygon.is_valid(), contours))

                    for idx, cnt in enumerate(contours):
                        region_id = "_region%04d" % idx

                        self._add_TextRegion(
                            page,
                            page_image,
                            page_xywh,
                            page_id,
                            cnt.polygon.points,
                            region_id
                        )

                else:
                    self.log.error(
                        "PAGE-XML does not support sub-element of type %s for element Page",
                        self.parameter['type']
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
                        page_xywh
                    )

                    # Convert PIL to cv2 (RGB):
                    region_image, alpha = pil_to_cv2_rgb(region_image)

                    # Get prediction for segment:
                    region_prediction = model.predict(region_image)

                    if self.parameter['type'] == "AlternativeImageType":
                        # Convert to cv2 binary image then to PIL:
                        region_prediction = cv2_to_pil_gray(region_prediction.to_binary_image(), alpha=alpha)

                        self._add_AlternativeImage(page_id, region, region_image, region_xywh, region_id, "")

                    elif self.parameter['type'] == "TextLineType":
                        # Find contours of prediction:
                        contours = Contour.from_image(region_prediction.img)

                        # Filter out child contours:
                        contours = list(filter(lambda cnt: not cnt.is_child(), contours))

                        # Filter out invalid polygons:
                        contours = list(filter(lambda cnt: cnt.polygon.is_valid(), contours))

                        for idx, cnt in enumerate(contours):
                            line_id = "_line%04d" % idx

                            self._add_TextLine(
                                page_id,
                                region,
                                region_image,
                                region_xywh,
                                region_id,
                                cnt.polygon.points,
                                line_id
                            )

                    else:
                        self.log.error(
                            "PAGE-XML does not support sub-element of type %s for element TextRegion",
                            self.parameter['type']
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
                            page_xywh
                        )

                        # Convert PIL to cv2 (RGB):
                        line_image, alpha = pil_to_cv2_rgb(line_image)

                        # Get prediction for segment:
                        line_prediction = model.predict(line_image)

                        if self.parameter['type'] == "AlternativeImageType":
                            # Convert to cv2 binary image then to PIL:
                            line_prediction = cv2_to_pil_gray(line_prediction.to_binary_image(), alpha=alpha)

                            self._add_AlternativeImage(page_id, line, line_image, line_xywh, region_id+line_id, "")

                        else:
                            self.log.error(
                                "PAGE-XML does not support sub-element of type %s for element TextLine",
                                self.parameter['type']
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
