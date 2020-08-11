German-Brazilian Newspapers (gbn)
=================================

This project aims at providing an [OCR-D](https://ocr-d.de/) compliant toolset for [optical layout recognition/analysis](https://en.wikipedia.org/wiki/Document_layout_analysis) on images of historical german-language documents published in Brazil during the 19th and 20th centuries, focusing on periodical publications.

Table of contents
=================

<!--ts-->
   * [German-Brazilian Newspapers (gbn)](#german-brazilian-newspapers-(gbn))
   * [Table of contents](#table-of-contents)
   * [About](#about)
   * [Tools (gbn.sbb)](#tools-(gbn.sbb))
      * [ocrd-gbn-sbb-predict](#ocrd-gbn-sbb-predict)
      * [ocrd-gbn-sbb-crop](#ocrd-gbn-sbb-crop)
      * [ocrd-gbn-sbb-binarize](#ocrd-gbn-sbb-binarize)
      * [ocrd-gbn-sbb-segment](#ocrd-gbn-sbb-segment)
   * [Library (gbn.lib)](#library-(gbn.lib))
   * [Models](#models)
   * [Recommended Workflow](#recommended-workflow)
<!--te-->

About
=====

Although there is a considerable amount of digitized brazilian-published german-language periodicals available online (e.g. the [*dbp digital* collection](https://dokumente.ufpr.br/en/dbpdigital.html) and the [*German-language periodicals* section of the Brazilian (National) Digital Library](http://memoria.bn.br/docreader/docmulti.aspx?bib=ger)), document image understanding of these prints is far from being optimal. While generic [OCR](https://en.wikipedia.org/wiki/Optical_character_recognition) solutions will work out of the box with typical everyday-life documents, it is a different story for historical newspapers like those due to several factors:

   * Complex layouts (still a challenge for mainstream OCR toolsets e.g. [ocropy](https://github.com/tmbarchive/ocropy) and [tesseract](https://github.com/tesseract-ocr/tesseract))
   * Degradation over time (e.g. stains, rips, erased ink) 
   * Poor scanning quality (e.g. lighting contrast)

In order to achieve better [full-text recognition](https://ocr-d.de/en/about) results on the target documents, this project relies on two building blocks: The [German-Brazilian Newspapers dataset](https://web.inf.ufpr.br/vri/databases/gbn/) and the [ocrd-sbb-textline-detector tool](https://github.com/qurator-spk/sbb_textline_detection). The first as a role-model for pioneering on layout analysis of german-brazilian documents (and also as a source of testing data) and the latter as a reference implementation of a robust layout analysis workflow for german-language documents. This project itself was forked from [ocrd-sbb-textline-detector](https://github.com/qurator-spk/sbb_textline_detection), aiming at replicating the original tool's functionality into several smaller modules and extending it for more powerful workflows.

Tools (gbn.sbb)
===============

ocrd-gbn-sbb-predict
--------------------

```json
{
 "executable": "ocrd-gbn-sbb-predict",
 "categories": [
  "Layout analysis"
 ],
 "description": "Classifies pixels of input images given a binary (two classes) model and store the prediction as the specified PAGE-XML content type",
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
   "description": "Path to Keras model to be used",
   "required": true,
   "cacheable": true
  },
  "shaping": {
   "type": "string",
   "description": "How the images must be processed in order to match the input shape of the model ('resize' for resizing to model shape and 'split' for splitting into patches)",
   "required": true,
   "enum": [
    "resize",
    "split"
   ]
  },
  "type": {
   "type": "string",
   "description": "PAGE-XML content type to be predicted",
   "required": true,
   "enum": [
    "AlternativeImageType",
    "BorderType",
    "TextRegionType",
    "TextLineType"
   ]
  },
  "operation_level": {
   "type": "string",
   "description": "PAGE-XML hierarchy level to operate on",
   "default": "page",
   "enum": [
    "page",
    "region",
    "line"
   ]
  }
 }
}
```

ocrd-gbn-sbb-crop
-----------------

```json
{
 "executable": "ocrd-gbn-sbb-crop",
 "categories": [
  "Image preprocessing",
  "Layout analysis"
 ],
 "description": "Crops the input page images by predicting the actual page surface and setting the PAGE-XML Border accordingly",
 "steps": [
  "preprocessing/optimization/cropping",
  "layout/analysis"
 ],
 "input_file_grp": [
  "OCR-D-IMG"
 ],
 "output_file_grp": [
  "OCR-D-CROP"
 ],
 "parameters": {
  "model": {
   "type": "string",
   "description": "Path to Keras model to be used",
   "required": true,
   "cacheable": true
  },
  "shaping": {
   "type": "string",
   "description": "How the images must be processed in order to match the input shape of the model ('resize' for resizing to model shape and 'split' for splitting into patches)",
   "default": "resize",
   "enum": [
    "resize",
    "split"
   ]
  }
 }
}
```

ocrd-gbn-sbb-binarize
---------------------

```json
{
 "executable": "ocrd-gbn-sbb-binarize",
 "categories": [
  "Image preprocessing",
  "Layout analysis"
 ],
 "description": "Binarizes the input page images by predicting their foreground pixels and saving it as a PAGE-XML AlternativeImage",
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
   "description": "Path to Keras model to be used",
   "required": true,
   "cacheable": true
  },
  "shaping": {
   "type": "string",
   "description": "How the images must be processed in order to match the input shape of the model ('resize' for resizing to model shape and 'split' for splitting into patches)",
   "default": "split",
   "enum": [
    "resize",
    "split"
   ]
  },
  "operation_level": {
   "type": "string",
   "description": "PAGE-XML hierarchy level to operate on",
   "default": "page",
   "enum": [
    "page",
    "region",
    "line"
   ]
  }
 }
}
```

ocrd-gbn-sbb-segment
--------------------

```json
{
 "executable": "ocrd-gbn-sbb-segment",
 "categories": [
  "Layout analysis"
 ],
 "description": "Segments the input page images by predicting the text regions and lines and setting the PAGE-XML TextRegion and TextLine accordingly",
 "steps": [
  "layout/segmentation/region",
  "layout/segmentation/line"
 ],
 "input_file_grp": [
  "OCR-D-DESKEW"
 ],
 "output_file_grp": [
  "OCR-D-SEG"
 ],
 "parameters": {
  "region_model": {
   "type": "string",
   "description": "Path to Keras model to be used for predicting text regions",
   "default": "",
   "cacheable": true
  },
  "region_shaping": {
   "type": "string",
   "description": "How the images must be processed in order to match the input shape of the model ('resize' for resizing to model shape and 'split' for splitting into patches)",
   "default": "split",
   "enum": [
    "resize",
    "split"
   ]
  },
  "line_model": {
   "type": "string",
   "description": "Path to Keras model to be used for predicting text lines",
   "required": true,
   "cacheable": true
  },
  "line_shaping": {
   "type": "string",
   "description": "How the images must be processed in order to match the input shape of the model ('resize' for resizing to model shape and 'split' for splitting into patches)",
   "default": "split",
   "enum": [
    "resize",
    "split"
   ]
  }
 }
}
```

Library (gbn.lib)
=================

This small library provides an abstraction layer that the OCR-D processors contained in this project should use for performing common image processing and deep learning routines. Those processors therefore should not directly access libraries like OpenCV, Numpy or Keras.

Check the source code files for detailed documentation on each class and function of the library.

Models
======

Currently the models being used are the ones provided by the [qurator team](https://github.com/qurator-spk). Models for binarization can be found [here](https://qurator-data.de/sbb_binarization/) and for cropping and segmentation [here](https://qurator-data.de/sbb_textline_detector/).

There are plans for extending the [GBN dataset](https://web.inf.ufpr.br/vri/databases/gbn/) with more degraded document pages as an attempt to train robust models in the near future.

Recommended Workflow
====================

The most generic and simple processing step implementations of [ocrd-sbb-textline-detector](https://github.com/qurator-spk/sbb_textline_detection) were not implemented since there are already tools that do effectively the same. The resizing to **2800 pixels** of height is performed through an [imagemagick wrapper for OCR-D (ocrd-im6convert)](https://github.com/OCR-D/ocrd_im6convert) and the deskewing through an [ocropy wrapper (ocrd-cis-ocropy)](https://github.com/cisocrgroup/ocrd_cis).

| Step  | Processor                 | Parameters |
| ----- | ------------------------- | ---------- |
| 1     | ocrd-im6convert           | { "output-format": "image/png", "output-options": "-geometry x2800" } |
| 2     | ocrd-gbn-sbb-crop         | { "model": "/path/to/model_page_mixed_best.h5", "shaping": "resize" }	|
| 3     | ocrd-gbn-sbb-binarize     | { "model": "/path/to/model_bin4.h5", "shaping": "split", "operation_level": "page" } |
| 4     | ocrd-cis-ocropy-deskew    | { "level-of-operation": "page" } |
| 5     | ocrd-gbn-sbb-segment      | { "region_model": "/path/to/model_strukturerkennung.h5", "region_shaping": "split", "line_model": "/path/to/model_textline_new.h5", "line_shaping": "split" }	|
