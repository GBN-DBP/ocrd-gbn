from qurator.sbb_textline_detector.common import init_session, load_model
from qurator.sbb_textline_detector.tool import OCRD_TOOL

import os
import math
import cv2
import numpy as np
from PIL import Image
from shapely import geometry

import ocrd_models.ocrd_page
from ocrd import Processor
from ocrd_modelfactory import page_from_file
from ocrd_models import OcrdFile
from ocrd_models.ocrd_page_generateds import AlternativeImageType, MetadataItemType, LabelsType, LabelType, TextRegionType, CoordsType
from ocrd_utils import concat_padded, getLogger, MIMETYPE_PAGE, crop_image, coordinates_for_segment, points_from_polygon

TOOL = 'ocrd-sbb-segment-region'

LOG = getLogger('processor.SbbSegmentRegion')
FALLBACK_FILEGRP_IMG = 'OCR-D-IMG-SEG'

class SbbSegmentRegion(Processor):

    def __init__(self, *args, **kwargs):
        kwargs['ocrd_tool'] = OCRD_TOOL['tools'][TOOL]
        kwargs['version'] = OCRD_TOOL['version']
        super(SbbSegmentRegion, self).__init__(*args, **kwargs)
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

        margin = int(0.1 * img_width_model)

        width_mid = img_width_model - 2 * margin
        height_mid = img_height_model - 2 * margin

        img = image / 255.0

        img_h = img.shape[0]
        img_w = img.shape[1]

        if img_h < img_height_model:
            canvas = np.ones((img_height_model, img_w, 3))
            canvas[:img_h, :, :] = img
            img = canvas
            img_h = img_height_model

        if img_w < img_width_model:
            canvas = np.ones((img_h, img_width_model, 3))
            canvas[:, :img_w, :] = img
            img = canvas
            img_w = img_width_model

        prediction_true = np.zeros((img_h, img_w, 3))

        mask_true = np.zeros((img_h, img_w))

        nxf = math.ceil(img_w / float(width_mid))
        nyf = math.ceil(img_h / float(height_mid))

        for i in range(nxf):
            for j in range(nyf):
                index_x_d = i * width_mid
                index_x_u = index_x_d + img_width_model

                index_y_d = j * height_mid
                index_y_u = index_y_d + img_height_model

                if index_x_u > img_w:
                    index_x_u = img_w
                    index_x_d = img_w - img_width_model
                if index_y_u > img_h:
                    index_y_u = img_h
                    index_y_d = img_h - img_height_model

                img_patch = img[index_y_d:index_y_u, index_x_d:index_x_u, :]

                label_p_pred = model.predict(img_patch.reshape(1, img_patch.shape[0], img_patch.shape[1], img_patch.shape[2]))

                seg = np.argmax(label_p_pred, axis=3)[0]
                seg_color = np.repeat(seg[:, :, np.newaxis], 3, axis=2)

                if i==0 and j==0:
                    seg_color = seg_color[0:seg_color.shape[0] - margin, 0:seg_color.shape[1] - margin, :]
                    seg = seg[0:seg.shape[0] - margin, 0:seg.shape[1] - margin]

                    mask_true[index_y_d + 0:index_y_u - margin, index_x_d + 0:index_x_u - margin] = seg
                    prediction_true[index_y_d + 0:index_y_u - margin, index_x_d + 0:index_x_u - margin,
                    :] = seg_color
                        
                elif i==nxf-1 and j==nyf-1:
                    seg_color = seg_color[margin:seg_color.shape[0] - 0, margin:seg_color.shape[1] - 0, :]
                    seg = seg[margin:seg.shape[0] - 0, margin:seg.shape[1] - 0]

                    mask_true[index_y_d + margin:index_y_u - 0, index_x_d + margin:index_x_u - 0] = seg
                    prediction_true[index_y_d + margin:index_y_u - 0, index_x_d + margin:index_x_u - 0,
                    :] = seg_color
                        
                elif i==0 and j==nyf-1:
                    seg_color = seg_color[margin:seg_color.shape[0] - 0, 0:seg_color.shape[1] - margin, :]
                    seg = seg[margin:seg.shape[0] - 0, 0:seg.shape[1] - margin]

                    mask_true[index_y_d + margin:index_y_u - 0, index_x_d + 0:index_x_u - margin] = seg
                    prediction_true[index_y_d + margin:index_y_u - 0, index_x_d + 0:index_x_u - margin,
                    :] = seg_color
                        
                elif i==nxf-1 and j==0:
                    seg_color = seg_color[0:seg_color.shape[0] - margin, margin:seg_color.shape[1] - 0, :]
                    seg = seg[0:seg.shape[0] - margin, margin:seg.shape[1] - 0]

                    mask_true[index_y_d + 0:index_y_u - margin, index_x_d + margin:index_x_u - 0] = seg
                    prediction_true[index_y_d + 0:index_y_u - margin, index_x_d + margin:index_x_u - 0,
                    :] = seg_color
                        
                elif i==0 and j!=0 and j!=nyf-1:
                    seg_color = seg_color[margin:seg_color.shape[0] - margin, 0:seg_color.shape[1] - margin, :]
                    seg = seg[margin:seg.shape[0] - margin, 0:seg.shape[1] - margin]

                    mask_true[index_y_d + margin:index_y_u - margin, index_x_d + 0:index_x_u - margin] = seg
                    prediction_true[index_y_d + margin:index_y_u - margin, index_x_d + 0:index_x_u - margin,
                    :] = seg_color
                        
                elif i==nxf-1 and j!=0 and j!=nyf-1:
                    seg_color = seg_color[margin:seg_color.shape[0] - margin, margin:seg_color.shape[1] - 0, :]
                    seg = seg[margin:seg.shape[0] - margin, margin:seg.shape[1] - 0]

                    mask_true[index_y_d + margin:index_y_u - margin, index_x_d + margin:index_x_u - 0] = seg
                    prediction_true[index_y_d + margin:index_y_u - margin, index_x_d + margin:index_x_u - 0,
                    :] = seg_color
                        
                elif i!=0 and i!=nxf-1 and j==0:
                    seg_color = seg_color[0:seg_color.shape[0] - margin, margin:seg_color.shape[1] - margin, :]
                    seg = seg[0:seg.shape[0] - margin, margin:seg.shape[1] - margin]

                    mask_true[index_y_d + 0:index_y_u - margin, index_x_d + margin:index_x_u - margin] = seg
                    prediction_true[index_y_d + 0:index_y_u - margin, index_x_d + margin:index_x_u - margin,
                    :] = seg_color
                        
                elif i!=0 and i!=nxf-1 and j==nyf-1:
                    seg_color = seg_color[margin:seg_color.shape[0] - 0, margin:seg_color.shape[1] - margin, :]
                    seg = seg[margin:seg.shape[0] - 0, margin:seg.shape[1] - margin]

                    mask_true[index_y_d + margin:index_y_u - 0, index_x_d + margin:index_x_u - margin] = seg
                    prediction_true[index_y_d + margin:index_y_u - 0, index_x_d + margin:index_x_u - margin,
                    :] = seg_color

                else:
                    seg_color = seg_color[margin:seg_color.shape[0] - margin, margin:seg_color.shape[1] - margin, :]
                    seg = seg[margin:seg.shape[0] - margin, margin:seg.shape[1] - margin]

                    mask_true[index_y_d + margin:index_y_u - margin, index_x_d + margin:index_x_u - margin] = seg
                    prediction_true[index_y_d + margin:index_y_u - margin, index_x_d + margin:index_x_u - margin,
                    :] = seg_color

        if img_h > image.shape[0]:
            prediction_true = prediction_true[:image.shape[0], :, :]

        if img_w > image.shape[1]:
            prediction_true = prediction_true[:, :image.shape[1], :]

        return prediction_true.astype(np.uint8)

    def segment_region(self, page_image):
        return self.predict(page_image, self.model)[:, :, 0] * 255

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
                page,
                page_id,
                feature_selector='binarized'
            )

            regions = page.get_TextRegion()

            for idx, region in enumerate(regions):
                # process region:
                region_image_pil, region_coords = self.workspace.image_from_segment(
                    region, page_image_pil, page_xywh, feature_selector='binarized,deskewed')

                if region_image_pil.mode == "LA":
                    alpha = region_image_pil.getchannel('A')

                    bg = Image.new("LA", region_image_pil.size, 255)
                    bg.paste(region_image_pil, mask=alpha)

                    region_image_pil = bg.convert('L')

                region_image_cv2_bin = np.array(region_image_pil, dtype=np.uint8)

                print("Region {}:".format(idx))
                region_image_cv2 = cv2.cvtColor(region_image_cv2_bin, cv2.COLOR_GRAY2BGR)

                self.kernel = np.ones((5, 5), np.uint8)

                self.session = init_session()
                self.model = load_model(self.parameter['model'])

                try:
                    pred = self.segment_region(region_image_cv2)
                    print("Region {} done".format(idx))
                except Exception as e:
                    pred = None
                    print("Region {} failed: {}".format(idx, e))

                self.session.close()

                if pred is not None:
                    region_image_pil = Image.fromarray(pred)

                    file_id = input_file.ID.replace(self.input_file_grp, self.image_grp)

                    if file_id == input_file.ID:
                        file_id = concat_padded(self.image_grp, n)

                    file_id += "LINE_MASK_%04d" % idx

                    self.workspace.save_image_file(
                        region_image_pil,
                        file_id,
                        page_id=page_id,
                        file_grp=self.image_grp
                    )

            continue

            page_image_cv2_bin = np.array(page_image_pil, dtype=np.uint8) * 255

            page_image_cv2 = cv2.cvtColor(page_image_cv2_bin, cv2.COLOR_GRAY2BGR)

            self.kernel = np.ones((5, 5), np.uint8)

            self.session = init_session()
            self.model = load_model(self.parameter['model'])

            pred = self.segment_region(page_image_cv2)

            self.session.close()

            page_image_pil = Image.fromarray(pred)

            file_id = input_file.ID.replace(self.input_file_grp, self.image_grp)

            if file_id == input_file.ID:
                file_id = concat_padded(self.image_grp, n)

            file_id += "LINE_MASK"

            self.workspace.save_image_file(
                page_image_pil,
                file_id,
                page_id=page_id,
                file_grp=self.image_grp
            )

            continue

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
