# gbn - German-Brazilian Newspapers

Collection of [OCR-D](https://ocr-d.de) compliant tools for layout analysis and segmentation of historical documents from the [German-Brazilian Newspapers dataset](https://web.inf.ufpr.br/vri/databases/gbn). Forked from *Staatsbibliothek zu Berlin*'s [Textline Detector](https://github.com/qurator-spk/sbb_textline_detection).

This project is still a `Work in Progress`, so there are still tools to implement and workflows to test in order to perform full-text recognition in the dataset, and the existing ones might be modified to best fulfill our needs.

## Tools

### ocrd-gbn-mask

Given two or more input file groups, being the last one the one that contains the masks for each page, the mask is applied for each page of each remaining file group. All files must be binary, including the masks and the input images, which must contain the `binarized` comment. Notice that for each input file group, its respective output file group must be also specified.

The background color, to which the pixels outside of the mask (black part of image) are set, can be defined through the parameter `bg_color` to either "black" or "white".

```json
{
 "executable": "ocrd-gbn-mask",
 "categories": [
  "Image preprocessing"
 ],
 "description": "Applies a mask (represented as a binary, black-and-white image) on the binarized input page",
 "steps": [
  "preprocessing/optimization"
 ],
 "input_file_grp": [
  "OCR-D-BIN"
 ],
 "output_file_grp": [
  "OCR-D-MASK"
 ],
 "parameters": {
  "bg_color": {
   "type": "string",
   "enum": [
    "black",
    "white"
   ],
   "default": "black",
   "description": "Background color, to be set to the pixels of the output page not belonging to the mask"
  }
 }
}
```

### ocrd-gbn-sbb-predict

Given an input file group and a path to one a model, runs it for each page and outputs the labels for each pixel as a binary image. This output file group can be used as e.g. the mask file group of `ocrd-gbn-mask`.

The path of the used model must be provided through the `model` parameter.

The method used for predicting can be specified through `prediction_method`. Those method are: `whole`, `patches` or `bordered_patches`. The first is from the [sbb-textline-detector](https://github.com/qurator-spk/sbb_textline_detection), which consists of resizing the image to the model dimensions and passing it to the model. The second is also from `sbb-textline-detector`, consisting of splitting the image in patches and passing them to the model. The last and third method is a variant of the second method, which applies a padding to the image so no overlapping of patches occurs, and also makes a border around the image so the routine of discarding the patches' borders is the same for all patches.

Lastly, there is a `operation_level` parameter, but currently only the `page` operation level is supported.

```json
{
 "executable": "ocrd-gbn-sbb-predict",
 "categories": [
  "Layout analysis"
 ],
 "description": "Predicts pixels of input image given a model and outputs the labels given to each pixel as a binary image",
 "steps": [
  "layout/analysis"
 ],
 "input_file_grp": [
  "OCR-D-IMG",
  "OCR-D-BIN"
 ],
 "output_file_grp": [
  "OCR-D-PREDICT"
 ],
 "parameters": {
  "model": {
   "type": "string",
   "format": "file",
   "cacheable": true,
   "description": "Path of model to be run"
  },
  "prediction_method": {
   "type": "string",
   "enum": [
    "whole",
    "patches",
    "bordered_patches"
   ],
   "default": "whole",
   "description": "How the image should be passed to the model (whole image or split in patches)"
  },
  "operation_level": {
   "type": "string",
   "enum": [
    "page",
    "region",
    "line"
   ],
   "default": "page",
   "description": "PAGE XML hierarchy level to operate on"
  }
 }
}
```

### ocrd-gbn-sbb-page-segment

`Work in Progress`

### ocrd-gbn-sbb-region-segment

`Work in Progress`

## Proposed Workflow

The following diagram describes the workflow we intend to use on the dataset. The underscored tools are the ones being implemented on this project.

![Workflow](workflow.png)
