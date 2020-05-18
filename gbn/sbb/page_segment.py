from gbn.tool import OCRD_TOOL

from ocrd import Processor
from ocrd_modelfactory import page_from_file
from ocrd_models.ocrd_page import to_xml
from ocrd_models.ocrd_page_generateds import AlternativeImageType, CoordsType, LabelsType, LabelType, MetadataItemType, TextRegionType
from ocrd_utils import concat_padded, coordinates_for_segment, getLogger, MIMETYPE_PAGE, points_from_polygon

import os.path
import cv2
import numpy as np
import PIL.Image
from shapely import geometry

TOOL = "ocrd-gbn-sbb-page-segment"

LOG = getLogger("processor.PageSegment")
FALLBACK_FILEGRP_IMG = "OCR-D-IMG-SEG" # DEBUG ONLY

class PageSegment(Processor):
    def __init__(self, *args, **kwargs):
        kwargs['ocrd_tool'] = OCRD_TOOL['tools'][TOOL]
        kwargs['version'] = OCRD_TOOL['version']
        super(PageSegment, self).__init__(*args, **kwargs)

        if hasattr(self, "input_file_grp"):
            try:
                self.input_file_grp, self.txtreg_grp, self.txtline_grp = self.input_file_grp.split(',')
            except ValueError:
                LOG.error("Three input groups required (input_group, txtreg_group, txtline_group)")
                quit()

            try:
                self.page_grp, self.image_grp = self.output_file_grp.split(',')
            except ValueError:
                self.page_grp = self.output_file_grp
                self.image_grp = FALLBACK_FILEGRP_IMG # DEBUG ONLY
                LOG.info("No output file group for images specified, falling back to '%s'", FALLBACK_FILEGRP_IMG)

    @property
    def txtreg_files(self):
        """
        List the text region prediction image files
        """
        return self.workspace.mets.find_files(fileGrp=self.txtreg_grp, pageId=self.page_id)

    @property
    def txtline_files(self):
        """
        List the text line prediction image files
        """
        return self.workspace.mets.find_files(fileGrp=self.txtline_grp, pageId=self.page_id)

    def textline_filter(self, boxes, predict_image):
        filtered = []

        for box in boxes:
            x0 = box[0]
            y0 = box[1]
            x1 = box[0] + box[2]
            y1 = box[1] + box[3]

            chunk = predict_image[y0:y1, x0:x1]

            area = chunk.shape[0] * chunk.shape[1]
            positives = np.count_nonzero(chunk[:, :] == 255)

            if positives / area >= self.parameter['min_textline_density']:
                filtered.append(box)

        return filtered

    def foreground_filter(self, boxes, page_image, predict_image):
        filtered = []

        for box in boxes:
            x0 = box[0]
            y0 = box[1]
            x1 = box[0] + box[2]
            y1 = box[1] + box[3]

            bin_chunk = page_image[y0:y1, x0:x1]
            pred_chunk = (predict_image[y0:y1, x0:x1] / 255).astype(np.bool_)

            area = bin_chunk[pred_chunk == False].shape[0]
            positives = np.count_nonzero(bin_chunk[pred_chunk == False] == 0)

            if positives / area <= self.parameter['max_foreground_density']:
                filtered.append(box)

        return filtered

    def segment_page(self, predict_image):
        regions = cv2.cvtColor((predict_image / 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        regions = cv2.erode(regions, self.kernel, iterations=3)
        regions = cv2.dilate(regions, self.kernel, iterations=4)

        text_class = (1, 1, 1)
        mask_texts = np.all(regions == text_class, axis=-1)

        image = np.repeat(mask_texts[:, :, np.newaxis], 3, axis=2) * 255
        image = image.astype(np.uint8)

        image = cv2.morphologyEx(image, cv2.MORPH_OPEN, self.kernel)
        image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, self.kernel)

        imgray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(imgray, 0, 255, 0)

        contours, hierarchy = cv2.findContours(thresh.copy(), cv2.cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        main_contours = list()

        jv = 0
        for c in contours:
            if len(c) < 3:  # A polygon cannot have less than 3 points
                continue

            polygon = geometry.Polygon([point[0] for point in c])
            area = polygon.area
            if area >= 0.00001 * np.prod(thresh.shape[:2]) and area <= 1 * np.prod(
                    thresh.shape[:2]) and hierarchy[0][jv][3] == -1 :
                main_contours.append(
                    np.array([ [point] for point in polygon.exterior.coords], dtype=np.uint))
            jv += 1

        boxes = []
        
        for jj in range(len(main_contours)):
            x, y, w, h = cv2.boundingRect(main_contours[jj])
            boxes.append([x, y, w, h])

        return boxes

    def process(self):
        for n, (input_file, txtreg_file, txtline_file) in enumerate(zip(self.input_files, self.txtreg_files, self.txtline_files)):
            LOG.info("Processing binary page image input file %i / %s", n, input_file)
            LOG.info("Processing text regions prediction image input file %i / %s", n, txtreg_file)
            LOG.info("Processing text lines prediction image input file %i / %s", n, txtline_file)

            # Create a new PAGE file from the input file:
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

            # Create a new PAGE file from text region prediction image file:
            txtreg_page_id = txtreg_file.pageId or txtreg_file.ID
            txtreg_pcgts = page_from_file(self.workspace.download_file(txtreg_file))
            txtreg_page = txtreg_pcgts.get_Page()

            # Get text regions prediction image from PAGE:
            txtreg, _, _ = self.workspace.image_from_page(
                txtreg_page,
                txtreg_page_id,
            )

            # Convert PIL image array to 8-bit grayscale then to uint8 Numpy array (for OpenCV):
            txtreg = np.array(txtreg.convert('L'), dtype=np.uint8)

            # Create a new PAGE file from the text line prediction image file:
            txtline_page_id = txtline_file.pageId or txtline_file.ID
            txtline_pcgts = page_from_file(self.workspace.download_file(txtline_file))
            txtline_page = txtline_pcgts.get_Page()

            # Get text lines prediction image from PAGE:
            txtline, _, _ = self.workspace.image_from_page(
                txtline_page,
                txtline_page_id,
            )

            # Convert PIL image array to 8-bit grayscale then to uint8 Numpy array (for OpenCV):
            txtline = np.array(txtline.convert('L'), dtype=np.uint8)

            # TODO: Unhardcode this:
            self.kernel = np.ones((5, 5), np.uint8)

            # Retrieve bounding boxes of segmented regions:
            boxes = self.segment_page(txtreg)

            # Apply textline density filter:
            boxes = self.textline_filter(boxes, txtline)

            # Apply foreground density filter:
            boxes = self.foreground_filter(boxes, page_image, txtline)

            # DEBUG ONLY
            txtline_mask = (txtline / 255).astype(np.bool_)
            bg_mask = (page_image / 255).astype(np.bool_)

            page_image = cv2.cvtColor(page_image, cv2.COLOR_GRAY2BGR)

            page_image[bg_mask == True] = np.array([0, 0, 0])
            page_image[txtline_mask == True] = np.array([255, 0, 0])
            page_image[bg_mask == False] = np.array([255, 255, 255])
            ############

            for region, box in enumerate(boxes):
                x0 = box[0]
                y0 = box[1]
                x1 = box[0] + box[2]
                y1 = box[1] + box[3]

                polygon = [
                    [x0, y0],
                    [x1, y0],
                    [x1, y1],
                    [x0, y1]
                ]

                # DEBUG ONLY
                cv2.rectangle(page_image, (x0, y0), (x1, y1), (0, 255, 0), 3)
                ############

                region_id = page_id + "_region%04d" % region

                # convert back to absolute (page) coordinates:
                region_polygon = coordinates_for_segment(polygon, page_image, page_xywh)

                # annotate result:
                page.add_TextRegion(
                    TextRegionType(
                        id=region_id,
                        Coords=CoordsType(
                            points=points_from_polygon(region_polygon)
                        )
                    )
                )

            # DEBUG ONLY
            page_image = PIL.Image.fromarray(cv2.cvtColor(page_image, cv2.COLOR_BGR2RGB))

            file_id = txtline_file.ID.replace(self.input_file_grp, self.image_grp)

            if file_id == txtline_file.ID:
                file_id = concat_padded(self.image_grp, n)

            self.workspace.save_image_file(
                page_image,
                file_id,
                page_id=page_id,
                file_grp=self.image_grp
            )
            ############

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

            # Get file ID of XML PAGE to be saved:
            file_id = input_file.ID.replace(self.input_file_grp, self.page_grp)

            if file_id == txtline_file.ID:
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
