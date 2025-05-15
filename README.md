# Oral Cancer Detection using U-Net and YOLOv8

This project focuses on detecting oral cancer through medical image analysis using a combination of deep learning models — **U-Net** for segmentation and **YOLOv8** for object detection and recognition. The dataset comprises annotated oral cavity images, enabling a two-step pipeline to identify lesions and classify the affected areas.

## Project Overview

The pipeline consists of:
1. **Segmentation with U-Net**: To isolate regions of interest (lesions, oral cavity).
2. **Detection with YOLOv8**: To localize and recognize the segmented regions.

## Objectives

- Develop a deep learning model that predicts the presence of oral cancer based on image data.
- Implement both segmentation and detection models.
- Compare model performance against alternative approaches.
- Enable future deployment of the solution in a clinical setting.

## Dataset

- **Source**: Clinical images of oral cavities.
- **Format**: Images in JPG, annotations in COCO format, metadata in CSV.
- **Classes**:
  - `0`: Lesion
  - `1`: Oral Cavity

> Dataset includes patient metadata like age, sex, diagnosis, and risk factors (smoking, alcohol, betel quid chewing).

## ⚙️ Model Details

### 1. Segmentation (U-Net)
- Input: Oral cavity images
- Output: Binary masks highlighting lesions and oral cavity
- Metrics: IoU, Dice Score

### 2. Detection (YOLOv8)
- Input: Original images with bounding boxes
- Output: Bounding boxes around lesions and oral cavities
- Metrics: mAP, Precision, Recall

## Future Work
- Model deployment using Streamlit or Flask.
- Support for video-based diagnosis.
- Expand dataset diversity (ethnic, age groups, lighting).
- Integrate clinical metadata for multimodal learning.

