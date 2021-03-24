import os

import ocrd
import ocrd_modelfactory
import ocrd_models.ocrd_page as ocrd_page
import ocrd_models.ocrd_page_generateds as ocrd_page_gends
import ocrd_utils

from gbn.tool import OCRD_TOOL


class OcrdGbnProcessor(ocrd.Processor):
    tool = "ocrd-gbn-processor"
    log = ocrd_utils.getLogger("processor.OcrdGbnProcessor")

    fallback_image_filegrp = "OCR-D-IMG-PROCESSOR"

    def __init__(self, *args, **kwargs):
        kwargs['ocrd_tool'] = OCRD_TOOL['tools'][self.tool]
        kwargs['version'] = OCRD_TOOL['version']
        super(OcrdGbnProcessor, self).__init__(*args, **kwargs)

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
                    "No output file group for images specified, "
                    "falling back to '%s'",
                    self.fallback_image_filegrp
                )

    def file_id(self, file_grp):
        file_id = self.input_file.ID.replace(self.input_file_grp, file_grp)

        if file_id == self.input_file.ID:
            file_id = ocrd_utils.concat_padded(file_grp, self.page_num)

        return file_id

    @property
    def page_file_id(self):
        return self.file_id(self.page_grp)

    @property
    def image_file_id(self):
        return self.file_id(self.image_grp)

    def _add_AlternativeImage(self, page_id, segment, segment_image,
                              segment_xywh, segment_id, comments):
        # Save image:
        file_path = self.workspace.save_image_file(
            segment_image,
            self.image_file_id+segment_id,
            page_id=page_id,
            file_grp=self.image_grp
        )

        # Add metadata about saved image:
        segment.add_AlternativeImage(
            ocrd_page_gends.AlternativeImageType(
                filename=file_path,
                comments=comments if not segment_xywh['features'] else
                segment_xywh['features'] + "," + comments
            )
        )

    def _set_Border(self, page, page_image, page_xywh, border_polygon):
        # Convert to absolute (page) coordinates:
        border_polygon = ocrd_utils.coordinates_for_segment(
            border_polygon,
            page_image,
            page_xywh
        )

        # Save border:
        page.set_Border(
            ocrd_page_gends.BorderType(
                Coords=ocrd_page_gends.CoordsType(
                    points=ocrd_utils.points_from_polygon(border_polygon)
                )
            )
        )

    def _set_PrintSpace(self, page, page_image, page_xywh,
                        print_space_polygon):
        # Convert to absolute (page) coordinates:
        border_polygon = ocrd_utils.coordinates_for_segment(
            print_space_polygon,
            page_image,
            page_xywh
        )

        # Save print space:
        page.set_PrintSpace(
            ocrd_page_gends.PrintSpaceType(
                Coords=ocrd_page_gends.CoordsType(
                    points=ocrd_utils.points_from_polygon(border_polygon)
                )
            )
        )

    def _add_Region(self, page, page_image, page_xywh, page_id,
                    region_class, region_polygon, region_id):
        # Convert to absolute (page) coordinates:
        region_polygon = ocrd_utils.coordinates_for_segment(
            region_polygon,
            page_image,
            page_xywh
        )

        add_Region = getattr(page, "add_" + region_class)

        # Save region:
        add_Region(
            ocrd_page_gends.TextRegionType(
                id=page_id + region_id,
                Coords=ocrd_page_gends.CoordsType(
                    points=ocrd_utils.points_from_polygon(region_polygon)
                )
            )
        )

    def _add_TextLine(self, page_id, region, region_image, region_xywh,
                      region_id, line_polygon, line_id):
        # Convert to absolute (page) coordinates:
        line_polygon = ocrd_utils.coordinates_for_segment(
            line_polygon,
            region_image,
            region_xywh
        )

        # Save text line:
        region.add_TextLine(
            ocrd_page_gends.TextLineType(
                id=page_id+region_id+line_id,
                Coords=ocrd_page_gends.CoordsType(
                    points=ocrd_utils.points_from_polygon(line_polygon)
                )
            )
        )

    def _add_metadata(self, pcgts):
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

    def _save_xml(self, pcgts, page_id):
        self.workspace.add_file(
            ID=self.page_file_id,
            file_grp=self.page_grp,
            pageId=page_id,
            mimetype=ocrd_modelfactory.MIMETYPE_PAGE,
            local_filename=os.join(
                self.output_file_grp, self.page_file_id
            ) + ".xml",
            content=ocrd_page.to_xml(pcgts)
        )
