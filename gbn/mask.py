from gbn.tool import OCRD_TOOL

from ocrd import Processor
from ocrd_modelfactory import page_from_file
from ocrd_models.ocrd_page import to_xml
from ocrd_models.ocrd_page_generateds import AlternativeImageType, LabelsType, LabelType, MetadataItemType
from ocrd_utils import concat_padded, getLogger, MIMETYPE_PAGE

import os.path
import cv2
import numpy as np
import PIL.Image

TOOL = "ocrd-gbn-mask"

LOG = getLogger("processor.Mask")
FALLBACK_FILEGRP_IMG = "OCR-D-IMG-MASK"

class Mask(Processor):
    def __init__(self, *args, **kwargs):
        kwargs['ocrd_tool'] = OCRD_TOOL['tools'][TOOL]
        kwargs['version'] = OCRD_TOOL['version']
        super(Mask, self).__init__(*args, **kwargs)

        if hasattr(self, "input_file_grp"):
            self.input_file_grps = self.input_file_grp.split(',')
            self.output_file_grps = self.output_file_grp.split(',')

            # Last input group must be the mask group:
            self.mask_grp = self.input_file_grps.pop()

            if not self.input_file_grps:
                LOG.error("At least two input groups required (page_group_0, ..., page_group_n, mask_group)")
                quit()

            # If output group number matches input group number:
            if len(self.output_file_grps) == len(self.input_file_grps):
                # Image group not specified:
                self.image_grp = FALLBACK_FILEGRP_IMG
                LOG.info("No output file group for images specified, falling back to '%s'", FALLBACK_FILEGRP_IMG)
            # If extra output group:
            elif len(self.output_file_grps) == len(self.input_file_grps) + 1:
                # Image group specified:
                self.image_grp = self.output_file_grps.pop()
            else:
                LOG.error("Input/Output group number mismatch (must be equal)")
                quit()

    @property
    def mask_files(self):
        """
        List the mask files
        """
        return self.workspace.mets.find_files(fileGrp=self.mask_grp, pageId=self.page_id)

    def mask_page(self, page_image, mask, bg_color):
        # TODO: Ensure the contour is completely filled

        # Fill canvas with background color:
        canvas = np.ones_like(page_image) * bg_color

        # Copy masked pixels of page image to canvas:
        canvas[mask == True] = page_image[mask == True]

        return canvas

    def process(self):
        for self.input_file_grp, self.output_file_grp in zip(self.input_file_grps, self.output_file_grps):
            for n, (input_file, mask_file) in enumerate(zip(self.input_files, self.mask_files)):
                LOG.info("Processing input file %i / %s", n, input_file)

                # Create a new PAGE file from the page image input file:
                page_id = input_file.pageId or input_file.ID
                pcgts = page_from_file(self.workspace.download_file(input_file))
                page = pcgts.get_Page()

                # Get image from PAGE (must have been binarized):
                page_image, page_xywh, _ = self.workspace.image_from_page(
                    page,
                    page_id,
                    feature_selector="binarized"
                )

                # Convert PIL image array to 8-bit grayscale then to uint8 Numpy array (for OpenCV):
                page_image = np.array(page_image.convert('L'), dtype=np.uint8)

                # Create a new PAGE file from the mask file:
                mask_page_id = mask_file.pageId or mask_file.ID
                mask_pcgts = page_from_file(self.workspace.download_file(mask_file))
                mask_page = mask_pcgts.get_Page()

                # Get image from PAGE:
                mask, _, _ = self.workspace.image_from_page(
                    mask_page,
                    mask_page_id
                )

                # Convert PIL image array to 1-bit grayscale then to boolean Numpy array (for OpenCV):
                mask = np.array(mask.convert('1'), dtype=np.bool_)

                # Apply mask on page image:
                masked = self.mask_page(page_image, mask, 0 if self.parameter['bg_color'] == "black" else 255)

                # Convert Numpy array (OpenCV image) to PIL image array then to 1-bit grayscale:
                masked = PIL.Image.fromarray(masked).convert('1')

                # Add metadata about this operation:
                metadata = pcgts.get_Metadata()
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

                # Get file ID of image to be saved:
                file_id = input_file.ID.replace(self.input_file_grp, self.image_grp)

                if file_id == input_file.ID:
                    file_id = concat_padded(self.image_grp, n)

                # Concatenate original file ID and mask file ID to it:
                file_id += "_" + input_file.ID + "_" + mask_file.ID

                # Save image:
                file_path = self.workspace.save_image_file(
                    masked,
                    file_id,
                    page_id=page_id,
                    file_grp=self.image_grp
                )

                # Add metadata about saved image:
                page.add_AlternativeImage(
                    AlternativeImageType(
                        filename=file_path,
                        comments=page_xywh['features']
                    )
                )

                # Get file ID of XML PAGE to be saved:
                file_id = input_file.ID.replace(self.input_file_grp, self.output_file_grp)

                if file_id == input_file.ID:
                    file_id = concat_padded(self.output_file_grp, n)

                # Save XML PAGE:
                self.workspace.add_file(
                     ID=file_id,
                     file_grp=self.output_file_grp,
                     pageId=page_id,
                     mimetype=MIMETYPE_PAGE,
                     local_filename=os.path.join(self.output_file_grp, file_id)+".xml",
                     content=to_xml(pcgts)
                )
