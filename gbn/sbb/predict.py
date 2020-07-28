from gbn.lib.dl import predict
from gbn.lib.struct import image
from gbn.lib.util import pil_to_cv2_rgb, cv2_to_pil_gray
from gbn.tool import OCRD_TOOL

from ocrd import Processor
from ocrd_modelfactory import page_from_file
from ocrd_models.ocrd_page import to_xml
from ocrd_models.ocrd_page_generateds import AlternativeImageType, BorderType, CoordsType, LabelsType, LabelType, MetadataItemType, TextLineType, TextRegionType
from ocrd_utils import concat_padded, coordinates_for_segment, getLogger, MIMETYPE_PAGE, points_from_polygon

from os.path import realpath, join

TOOL = "ocrd-gbn-sbb-predict"
LOG = getLogger("processor.OcrdGbnSbbPredict")

class OcrdGbnSbbPredict(Processor):
    def __init__(self, *args, **kwargs):
        kwargs['ocrd_tool'] = OCRD_TOOL['tools'][TOOL]
        kwargs['version'] = OCRD_TOOL['version']
        super(OcrdGbnSbbPredict, self).__init__(*args, **kwargs)

        if hasattr(self, "output_file_grp"):
            self.page_grp = self.output_file_grp

    def _predict_segment(self, segment, segment_image, segment_xywh, segment_suffix, predictor):
        # Convert PIL to cv2 (RGB):
        segment_image, _ = pil_to_cv2_rgb(segment_image)

        # Get labels per-pixel and map them to grayscale:
        segment_prediction = predictor.predict(segment_image) * 255

        # Wrap prediction into image object:
        segment_prediction = image(segment_image)

        # Find contours of prediction:
        contours = segment_prediction.analyse_contours()

        # Filter out child contours:
        contours = list(filter(lambda x: not x.is_child(), contours))

        # Filter out invalid polygons:
        contours = list(filter(lambda x: x.is_polygon(), contours))

        if self.parameter['type'] == "BorderType":
            # Get areas of all contours:
            areas = [cnt.area for cnt in contours]

            # Get largest contour:
            cnt = contours[areas.index(max(areas))]

            # Convert to absolute (page) coordinates:
            polygon = coordinates_for_segment(cnt.points, segment_image, segment_xywh)

            # Save border:
            segment.set_Border(
                BorderType(
                    Coords=CoordsType(
                        points=points_from_polygon(polygon)
                    )
                )
            )
        elif self.parameter['type'] == "TextRegionType":
            for idx, cnt in enumerate(contours):
                # Convert to absolute (page) coordinates:
                polygon = coordinates_for_segment(cnt.points, segment_image, segment_xywh)

                region_suffix = "_region%04d" % idx

                # Save text region:
                segment.add_TextRegion(
                    TextRegionType(
                        id=self.page_id+segment_suffix+region_suffix,
                        Coords=CoordsType(
                            points=points_from_polygon(polygon)
                        )
                    )
                )
        elif self.parameter['type'] == "TextLineType":
            for idx, cnt in enumerate(contours):
                # Convert to absolute (page) coordinates:
                polygon = coordinates_for_segment(cnt.points, segment_image, segment_xywh)

                line_suffix = "_line%04d" % idx

                # Save text line:
                segment.add_TextLine(
                    TextLineType(
                        id=self.page_id+segment_suffix+line_suffix,
                        Coords=CoordsType(
                            points=points_from_polygon(polygon)
                        )
                    )
                )

    def process(self):
        # Ensure path to model is absolute:
        self.parameter['model'] = realpath(self.parameter['model'])

        # Construct predictor object:
        predictor = predict(self.parameter['model'], self.parameter['shaping'])

        for (self.n, self.input_file) in enumerate(self.input_files):
            LOG.info("Processing input file: %i / %s", self.n, self.input_file)

            # Create a new PAGE file from the input file:
            self.page_id = self.input_file.pageId or self.input_file.ID
            self.pcgts = page_from_file(self.workspace.download_file(self.input_file))
            self.page = self.pcgts.get_Page()

            if self.parameter['operation_level'] == "page":
                # Get image from PAGE:
                page_image, page_xywh, _ = self.workspace.image_from_page(
                    self.page,
                    self.page_id
                )

                # Perform prediction:
                self._predict_segment(
                    self.page,
                    page_image,
                    page_xywh,
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
                    region_image, region_xywh = self.workspace.image_from_segment(
                        region,
                        page_image,
                        page_xywh
                    )

                    # Perform prediction:
                    self._predict_segment(
                        region,
                        region_image,
                        region_xywh,
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
                        line_image, line_xywh = self.workspace.image_from_segment(
                            line,
                            page_image,
                            page_xywh
                        )

                        # Perform prediction:
                        self._predict_segment(
                            line,
                            line_image,
                            line_xywh,
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
