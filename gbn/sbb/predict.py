from gbn.tool import OCRD_TOOL
from gbn.sbb.common import init_session, load_model

from ocrd import Processor
from ocrd_modelfactory import page_from_file
from ocrd_models import OcrdFile
from ocrd_models.ocrd_page import to_xml
from ocrd_models.ocrd_page_generateds import AlternativeImageType, MetadataItemType, LabelsType, LabelType
from ocrd_utils import concat_padded, getLogger, MIMETYPE_PAGE

import os.path
import cv2
import numpy as np
import PIL

TOOL = 'ocrd-gbn-sbb-predict'

LOG = getLogger('processor.Predict')
FALLBACK_FILEGRP_IMG = 'OCR-D-IMG-PREDICT'

class Predict(Processor):

    def __init__(self, *args, **kwargs):
        kwargs['ocrd_tool'] = OCRD_TOOL['tools'][TOOL]
        kwargs['version'] = OCRD_TOOL['version']
        super(Predict, self).__init__(*args, **kwargs)
        if hasattr(self, 'output_file_grp'):
            try:
                self.page_grp, self.image_grp = self.output_file_grp.split(',')
            except ValueError:
                self.page_grp = self.output_file_grp
                self.image_grp = FALLBACK_FILEGRP_IMG
                LOG.info("No output file group for images specified, falling back to '%s'", FALLBACK_FILEGRP_IMG)

    def predict(self, image, model):
        # Get model input dimensions:
        model_h = model.layers[-1].output_shape[1]
        model_w = model.layers[-1].output_shape[2]

        border = int(0.1 * model_w)

        # Get patch dimensions (model input - borders):
        patch_h = model_h - 2 * border
        patch_w = model_w - 2 * border

        # Map pixels from [0,1] (binary) to [0,255] (grayscale):
        image = image / 255.0

        # Get original image dimensions:
        img_h = image.shape[0]
        img_w = image.shape[1]

        # Get padding for splitting image into equal-sized patches:
        padding_h = patch_h - (img_h % patch_h)
        padding_w = patch_w - (img_w % patch_w)

        # Apply padding:
        padded_img = cv2.copyMakeBorder(image, 0, padding_h, 0, padding_w, cv2.BORDER_CONSTANT, (255, 255, 255))

        # Make border of image:
        bordered_img = cv2.copyMakeBorder(padded_img, border, border, border, border, cv2.BORDER_CONSTANT, (255, 255, 255))

        # Get number of patches per dimension:
        nyf = int(padded_img.shape[0] / patch_h)
        nxf = int(padded_img.shape[1] / patch_w)

        # Predicted unbordered patches must be written to the canvas:
        canvas = np.zeros((padded_img.shape[0], padded_img.shape[1]))

        for i in range(nxf):
            for j in range(nyf):
                xd = i * patch_w
                xu = xd + (border + patch_w + border)

                yd = j * patch_h
                yu = yd + (border + patch_h + border)

                # Get patch with border:
                bordered_patch = bordered_img[yd:yu, xd:xu]

                # Make prediction:
                pred = model.predict(bordered_patch.reshape(1, bordered_patch.shape[0], bordered_patch.shape[1], bordered_patch.shape[2]))

                # Reshape prediction into a 2-dimensional array with a single color channel (binary):
                pred = np.argmax(pred, axis=3)[0]

                # Remove borders around predicted patch and add it to canvas:
                canvas[yd:yd+patch_h, xd:xd+patch_w] = pred[border:border+patch_h, border:border+patch_w]

        # Remove padding and return prediction image:
        return canvas[:img_h, :img_w].astype(np.uint8)

    def process(self):
        for (n, input_file) in enumerate(self.input_files):
            LOG.info("Processing input file: %i / %s", n, input_file)

            # Create a new PAGE file from the input file:
            page_id = input_file.pageId or input_file.ID
            pcgts = page_from_file(self.workspace.download_file(input_file))
            page = pcgts.get_Page()

            # Get image from PAGE (must have been binarized):
            page_image, page_xywh, _ = self.workspace.image_from_page(page, page_id, feature_selector="binarized")

            # Convert PIL image array to Numpy array (for OpenCV) and map pixels to grayscale:
            page_image = np.array(page_image, dtype=np.uint8) * 255

            # TODO: Model requires 3 channels, while 1 would be enough since the input image is binary:
            def triplicate_channel(img):
                res = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
                res[:, :, 0] = img
                res[:, :, 1] = img
                res[:, :, 2] = img
                return res

            page_image = triplicate_channel(page_image)

            session = init_session()
            model = load_model(self.parameter['model'])

            # Get labels per-pixel and map them to grayscale:
            predict_image = self.predict(page_image, model) * 255

            session.close()

            # Convert OpenCV image array (Numpy) to PIL image array:
            predict_image = PIL.Image.fromarray(predict_image)

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

            file_id += "_" + os.path.splitext(os.path.basename(self.parameter['model']))[0]

            # Save image:
            file_path = self.workspace.save_image_file(
                predict_image,
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
            file_id = input_file.ID.replace(self.input_file_grp, self.page_grp)

            if file_id == input_file.ID:
                file_id = concat_padded(self.page_grp, n)

            # Save XML PAGE:
            self.workspace.add_file(
                 ID=file_id,
                 file_grp=self.output_file_grp,
                 pageId=page_id,
                 mimetype=MIMETYPE_PAGE,
                 local_filename=os.path.join(self.output_file_grp, file_id)+".xml",
                 content=to_xml(pcgts)
            )
