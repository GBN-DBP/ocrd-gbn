from qurator.sbb_textline_detector.common import init_session, load_model
from qurator.sbb_textline_detector.tool import OCRD_TOOL

import os
import cv2
import numpy as np
from PIL import Image

import ocrd_models.ocrd_page
from ocrd import Processor
from ocrd_modelfactory import page_from_file
from ocrd_models import OcrdFile
from ocrd_models.ocrd_page_generateds import AlternativeImageType, MetadataItemType, LabelsType, LabelType, BorderType, CoordsType
from ocrd_utils import concat_padded, getLogger, MIMETYPE_PAGE, crop_image, coordinates_for_segment, points_from_polygon

TOOL = 'ocrd-sbb-crop'

LOG = getLogger('processor.SbbCrop')
FALLBACK_FILEGRP_IMG = 'OCR-D-IMG-CROP'

class SbbCrop(Processor):

    def __init__(self, *args, **kwargs):
        kwargs['ocrd_tool'] = OCRD_TOOL['tools'][TOOL]
        kwargs['version'] = OCRD_TOOL['version']
        super(SbbCrop, self).__init__(*args, **kwargs)
        if hasattr(self, 'output_file_grp'):
            try:
                self.page_grp, self.image_grp = self.output_file_grp.split(',')
            except ValueError:
                self.page_grp = self.output_file_grp
                self.image_grp = FALLBACK_FILEGRP_IMG
                LOG.info("No output file group for images specified, falling back to '%s'", FALLBACK_FILEGRP_IMG)

    def scale_image(self, image):
        if image.shape[0] < 2500:
            height = 2800
            width = int(height * image.shape[1] / float(image.shape[0]))
        else:
            height = int(image.shape[0] * 1.2) # 6500 (?)
            width = int(height * image.shape[1] / float(image.shape[0]))

        self.scale_y = image.shape[0] / float(height)
        self.scale_x = image.shape[1] / float(width)

        return cv2.resize(image, (width, height), interpolation=cv2.INTER_NEAREST)

    def predict(self, image, model):
        img_height_model = model.layers[-1].output_shape[1]
        img_width_model = model.layers[-1].output_shape[2]

        img = cv2.resize(image / 255.0, (img_width_model, img_height_model), interpolation=cv2.INTER_NEAREST)

        # Encapsulate image array inside a single-element array:
        label_p_pred = model.predict(img.reshape(1, img.shape[0], img.shape[1], img.shape[2]))

        seg = np.argmax(label_p_pred, axis=3)[0]
        seg_color = np.repeat(seg[:, :, np.newaxis], 3, axis=2)

        return cv2.resize(seg_color, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST).astype(np.uint8)

    def crop_page(self, page_image):
        # TODO: Get from binarized image
        imgray = cv2.cvtColor(self.predict(page_image, self.model), cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(imgray, 0, 255, 0)

        thresh = cv2.dilate(thresh, self.kernel, iterations=6)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        cnt_size = np.array([cv2.contourArea(contours[j]) for j in range(len(contours))])

        cnt = contours[np.argmax(cnt_size)]

        x, y, w, h = cv2.boundingRect(cnt)

        x0 = x
        x1 = x + w
        y0 = y
        y1 = y + h

        return x0, x1, y0, y1

    def process(self):
        for (n, input_file) in enumerate(self.input_files):
            page_id = input_file.pageId or input_file.ID
            LOG.info("INPUT FILE %i / %s", n, input_file)

            # Process the files
            try:
                os.mkdir(self.output_file_grp)
            except FileExistsError:
                pass

            # Create a new PAGE file from the input file
            pcgts = page_from_file(self.workspace.download_file(input_file))
            page = pcgts.get_Page()

            page_image_pil, page_xywh, page_image_info = self.workspace.image_from_page(
                page, page_id, feature_filter='cropped')

            page_image_cv2 = cv2.cvtColor(np.array(page_image_pil), cv2.COLOR_RGB2BGR)

            #scaled_image = self.scale_image(page_image_cv2)

            self.kernel = np.ones((5, 5), np.uint8)

            self.session = init_session()
            self.model = load_model(self.parameter['model'])

            #x0, x1, y0, y1 = self.crop_page(scaled_image)
            x0, x1, y0, y1 = self.crop_page(page_image_cv2)

            self.session.close()

            #scaled_x0 = int(x0 * self.scale_x)
            #scaled_x1 = int(x1 * self.scale_x)
            #scaled_y0 = int(y0 * self.scale_y)
            #scaled_y1 = int(y1 * self.scale_y)

            #box = (scaled_x0, scaled_y0, scaled_x1, scaled_y1)
            box = (x0, y0, x1, y1)

            #polygon = [
                #[scaled_x0, scaled_y0],
                #[scaled_x1, scaled_y0],
                #[scaled_x1, scaled_y1],
                #[scaled_x0, scaled_y1]
            #]

            polygon = [
                [x0, y0],
                [x1, y0],
                [x1, y1],
                [x0, y1]
            ]

            border_polygon = coordinates_for_segment(polygon, page_image_pil, page_xywh)
            border_points = points_from_polygon(border_polygon)
            brd = BorderType(Coords=CoordsType(border_points))

            page.set_Border(brd)

            cropped_page_image_pil = crop_image(page_image_pil, box=box)

            # Save metadata about this operation
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

            file_id = input_file.ID.replace(self.input_file_grp, self.image_grp)

            if file_id == input_file.ID:
                file_id = concat_padded(self.image_grp, n)

            file_path = self.workspace.save_image_file(
                cropped_page_image_pil,
                file_id,
                page_id=page_id,
                file_grp=self.image_grp
            )

            page.add_AlternativeImage(
                AlternativeImageType(
                    filename=file_path,
                    comments=page_xywh['features']+",cropped"
                )
            )

            file_id = input_file.ID.replace(self.input_file_grp, self.page_grp)

            if file_id == input_file.ID:
                file_id = concat_padded(self.page_grp, n)

            self.workspace.add_file(
                 ID=file_id,
                 file_grp=self.output_file_grp,
                 pageId=page_id,
                 mimetype=MIMETYPE_PAGE,
                 local_filename=os.path.join(self.output_file_grp, file_id)+".xml",
                 content=ocrd_models.ocrd_page.to_xml(pcgts)
            )
