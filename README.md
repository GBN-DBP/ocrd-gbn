German-Brazilian Newspapers (gbn)
=================================

Collection of [OCR-D](https://ocr-d.de/en/) compliant tools for layout analysis and segmentation of historical german-language documents published in brazilian territory during the 19th and 20th centuries, with emphasis in periodicals. 

Even though there is a huge volume of digitized documents of this kind (e.g. the [Dokumente project](https://dokumente.ufpr.br/en/index.html) and the [Brazilian National Library](http://memoria.bn.br/docreader/docmulti.aspx?bib=ger)), they are all split into several databases and usually with none or bad **OCR**. The latter is due mostly to three factors that influence mainly the [layout analysis](https://ocr-d.de/en/workflows#layout-analysis) step of text recognition:

   * Complex layouts (still a challenge for mainstream **OCR** toolsets e.g. [ocropy](https://github.com/tmbarchive/ocropy) and [tesseract](https://github.com/tesseract-ocr/tesseract))
   * Degradation over time (e.g. stains, rips, erased ink) 
   * Poor scanning quality (e.g. lighting contrast)

This project aims to provide tools for better **layout analysis** (and therefore [full-text recognition](https://ocr-d.de/en/about)) on those documents and hopefully help building (a) searchable collection(s) of german-brazilian newspapers, making things easier and simpler for future research on german colonization in Brazil.

Table of contents
=================

<!--ts-->
   * [German-Brazilian Newspapers (gbn)](#german-brazilian-newspapers-(gbn))
   * [Table of contents](#table-of-contents)
   * [Overview](#overview)
   * [Library (gbn.lib)](#library-(gbn.lib))
      * [extract](#extract)
      * [predict](#predict)
      * [util](#util)
   * [Tools](#tools)
      * [ocrd-gbn-sbb-predict](#ocrd-gbn-sbb-predict)
      * [ocrd-gbn-sbb-binarize](#ocrd-gbn-sbb-binarize)
      * [ocrd-gbn-sbb-crop](#ocrd-gbn-sbb-crop)
      * [ocrd-gbn-sbb-segment](#ocrd-gbn-sbb-segment)
   * [Models](#models)
   * [Recommended Workflow](#recommended-workflow)
<!--te-->

Overview
========

This project is based on [ocrd-sbb-textline-detector](https://github.com/qurator-spk/sbb_textline_detection). While I was studying the available **OCR** solutions and the [OCR-D framework](https://ocr-d.de/en/) came to my knowledge, I took a small set of considerably degraded newspaper pages and started playing around with the available tools. This tool in particular caught my attention because, besides some **false negative** and **undersegmentation** issues when detecting the text lines, the overall result was considerably better than the ones obtained through the previously mentioned mainstream solutions.

Since the original tool is monolithical by design, enclosing several [processing steps](https://ocr-d.de/en/workflows) in a single command line, not much could be done for improving the results through parameter tuning and switching its processing steps by other [OCR-D](https://ocr-d.de/en/) modules was not possible. Therefore, this project was created with the intent of replicating the functionality of the [ocrd-sbb-textline-detector](https://github.com/qurator-spk/sbb_textline_detection) into several smaller tools, allowing a more modular and customizable workflow.

Library (gbn.lib)
=================

This project went for a more **object-oriented** architecture than the original implementation. The most used routines are stored as objects and functions in a small library, and the processors do all the deep learning and image processing by interfacing with them.

extract
-------

Stores the **Extracting** class. It is used to *extract* characteristics of images, containing methods for contours and bounding boxes analysis/manipulation/filtering.

It should be interfaced with by constructing an object for each image to be analysed. There are several methods for filtering, merging and splitting bounding boxes, analysing distribution of foreground pixels and others that are used mainly for segmentation.

predict
-------

Stores the **Predicting** class. It is used to *predict* the labels of each pixel of an image given a model.

It should be interfaced with by constructing an object for each model to be run, then performing the prediction for an image through the **Predicting.predict** method. All the model loading and splitting-into-patches operations are handled internally.

util
----

Stores generic image processing and workspace handling functions that are used by nearly every tool (e.g. converting image formats).

Tools
=====

ocrd-gbn-sbb-predict
--------------------

Applies a per-pixel binary (positive/negative labels) deep learning model to the input images. Outputs binary images where white means **positive** and black means **negative**. The path of the model file must be provided through the **model** parameter, and the algorithm used for predicting the labels through the **prediction_algorithm** parameter. The possible values for the latter are:

   * **whole_image**: The input image is resized to the model dimensions, the prediction is performed then converted into an image and resized to the original input image dimensions. This algorithm is used in [ocrd-gbn-sbb-crop](#ocrd-gbn-sbb-crop).
   * **sbb_patches**: The input image is split into patches which have the same dimensions as the model. The patches are extracted through a sliding window with a small margin in the inner sides of the image. If part of the window is out of the bounds of the image, it is slidden back until it reaches the borders of the image, overlapping parts that were already predicted. This is the algorithm implemented originally in [ocrd-sbb-textline-detector](https://github.com/qurator-spk/sbb_textline_detection).
   * **gbn_patches**: The input image is split into patches which have the same dimensions as the model. A padding is applied around the image so the windows do not need to be slidden back as in the original implementation, avoiding redundancy. The margin from the original method is also not applied. This algorithm is faster given the lack of redundancy and with no significant drawbacks. It also supports **region-level** predicting, unlike **sbb_patches**, since the padding step makes it possible to predict images originally smaller than the model.

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
  "prediction_algorithm": {
   "type": "string",
   "enum": [
    "whole_image",
    "sbb_patches",
    "gbn_patches"
   ],
   "default": "gbn_patches",
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

ocrd-gbn-sbb-binarize
---------------------

Binarizes pages using deep learning, as in [sbb_binarize](https://github.com/qurator-spk/sbb_binarization). Supports the same parameters as [ocrd-gbn-sbb-predict](#ocrd-gbn-sbb-predict).

```json
{
 "executable": "ocrd-gbn-sbb-binarize",
 "categories": [
  "Image preprocessing",
  "Layout analysis"
 ],
 "description": "Binarizes the input images using deep learning",
 "steps": [
  "preprocessing/optimization/binarization",
  "layout/analysis"
 ],
 "input_file_grp": [
  "OCR-D-IMG"
 ],
 "output_file_grp": [
  "OCR-D-BIN"
 ],
 "parameters": {
  "model": {
   "type": "string",
   "format": "file",
   "cacheable": true,
   "description": "Path of model to be run"
  },
  "prediction_algorithm": {
   "type": "string",
   "enum": [
    "whole_image",
    "sbb_patches",
    "gbn_patches"
   ],
   "default": "gbn_patches",
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

ocrd-gbn-sbb-crop
-----------------

Crops pages using deep learning. Originally, the cropping step consisted of extracting the bounding box of the positive part of the prediction and slicing the page on those coordinates. On this implementation, the prediction is used to mask the region of interest of the binarized page. The results of the latter are better than the first on ripped pages, since most of the area outside the actual page is masked out. Supports the same parameters as [ocrd-gbn-sbb-predict](ocrd-gbn-sbb-predict).

```json
{
 "executable": "ocrd-gbn-sbb-crop",
 "categories": [
  "Image preprocessing",
  "Layout analysis"
 ],
 "description": "Crops the input images using deep learning",
 "steps": [
  "preprocessing/optimization/cropping",
  "layout/analysis"
 ],
 "input_file_grp": [
  "OCR-D-BIN"
 ],
 "output_file_grp": [
  "OCR-D-CROP"
 ],
 "parameters": {
  "model": {
   "type": "string",
   "format": "file",
   "cacheable": true,
   "description": "Path of model to be run"
  },
  "prediction_algorithm": {
   "type": "string",
   "enum": [
    "whole_image",
    "sbb_patches",
    "gbn_patches"
   ],
   "default": "whole_image",
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

ocrd-gbn-sbb-segment
--------------------

Segments pages given the deep learning predictions of text regions and text lines. The **page-level** routine is very similar to the [ocrd-sbb-textline-detector](https://github.com/qurator-spk/sbb_textline_detection), consisting of combining text region and text line predictions to define the regions boundaries. Some extra steps were implemented e.g. to filter out false positives, merge overlapping bounding boxes and split distinct regions predicted as a single one by the model. The **region-level** routine, on the other hand, was written specifically for this project, and consists of extracting the text lines from the predictions for each region.

```json
{
 "executable": "ocrd-gbn-sbb-segment",
 "categories": [
  "Layout analysis"
 ],
 "description": "Segments text regions using deep learning",
 "steps": [
  "layout/segmentation"
 ],
 "input_file_grp": [
  "OCR-D-DESKEW"
 ],
 "output_file_grp": [
  "OCR-D-SEG"
 ],
 "parameters": {
  "min_particle_size": {
   "type": "number",
   "default": 1e-05,
   "description": "Minimum ratio of the total segment area for a particle to be considered text"
  },
  "max_particle_size": {
   "type": "number",
   "default": 1.0,
   "description": "Maximum ratio of the total segment area for a particle to be considered text"
  },
  "min_textline_density": {
   "type": "number",
   "default": 0.1,
   "description": "Minimum ratio of predicted text line pixels of a segment for it to be considered text"
  },
  "max_textline_density": {
   "type": "number",
   "default": 1.0,
   "description": "Maximum ratio of predicted text line pixels of a segment for it to be considered text"
  },
  "textregion_src": {
   "type": "string",
   "format": "file",
   "cacheable": true,
   "description": "Path of model to be run or input file group for text region predictions"
  },
  "textline_src": {
   "type": "string",
   "format": "file",
   "cacheable": true,
   "description": "Path of model to be run or input file group for text line predictions"
  },
  "textregion_algorithm": {
   "type": "string",
   "enum": [
    "whole_image",
    "sbb_patches",
    "gbn_patches"
   ],
   "default": "gbn_patches",
   "description": "How the image should be passed to the text region model"
  },
  "textline_algorithm": {
   "type": "string",
   "enum": [
    "whole_image",
    "sbb_patches",
    "gbn_patches"
   ],
   "default": "gbn_patches",
   "description": "How the image should be passed to the text line model"
  },
  "operation_level": {
   "type": "string",
   "enum": [
    "page",
    "region"
   ],
   "default": "page",
   "description": "PAGE XML hierarchy level to operate on"
  }
 }
}
```

Models
======

Currently the models being used are the ones provided by the [qurator team](https://github.com/qurator-spk). Models for binarization can be found [here](https://qurator-data.de/sbb_binarization/) and for cropping and segmentation [here](https://qurator-data.de/sbb_textline_detector/).

There are plans for building a training dataset composed of german-brazilian publications in the near future.

Recommended Workflow
====================

The most generic and simple processing step implementations of [ocrd-sbb-textline-detector](https://github.com/qurator-spk/sbb_textline_detection) were not implemented since there are already tools that do effectively the same. The resizing to **2800 pixels** of height is performed through an [imagemagick wrapper for OCR-D (ocrd-im6convert)](https://github.com/OCR-D/ocrd_im6convert) and the deskewing through an [ocropy wrapper (ocrd-cis-ocropy)](https://github.com/cisocrgroup/ocrd_cis).

| Step  | Processor                 | Parameters																										|
| ----- | ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1     | ocrd-im6convert           | { "output-format": "image/png", "output-options": "-geometry x2800" }																			|
| 2     | ocrd-gbn-sbb-binarize     | { "model": "/path/to/model_bin4.h5", "prediction_algorithm": "gbn_patches", "operation_level": "page" }															|
| 3     | ocrd-gbn-sbb-crop         | { "model": "/path/to/model_page_mixed_best.h5", "prediction_algorithm": "whole_image", "operation_level": "page" }													|
| 4     | ocrd-cis-ocropy-deskew    | { "level-of-operation": "page" }																								|
| 5     | ocrd-gbn-sbb-segment      | { "textregion_src": "/path/to/model_strukturerkennung.h5", "textregion_algorithm": "gbn_patches", "textline_src": "/path/to/model_textline_new.h5", "textline_algorithm": "gbn_patches", "operation_level": "page" }	|
| 6     | ocrd-cis-ocropy-binarize  | { "level-of-operation": "region" }																							|
| 7     | ocrd-cis-ocropy-deskew    | { "level-of-operation": "region" }																							|
| 8     | ocrd-gbn-sbb-segment      | { "textline_src": "/home/sulzbals/ocrd/models/sbb/model_textline_new.h5", "textline_algorithm": "gbn_patches", "operation_level": "region" }										|
