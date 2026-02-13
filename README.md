# Cilia-motion-phenotyping

Fully automated and interpretable framework for motion phenotype classification from high-speed microscopy videos.
Provides reproducible feature extraction and hierarchical machine learning tools for biomedical video analysis.

---

The framework implements a fully automated pipeline for high-speed video microscopy (HSVM) analysis, including YOLO-based cilia segmentation, dense optical flow, biomechanically interpretable feature extraction (vorticity, strain, motion-mask overlap, angular stability, and wave-based periodicity), and a hierarchical machine learning architecture for classifying ciliary motion phenotypes (immotile, normal, stiff, circular) in the context of Primary Ciliary Dyskinesia (PCD).

---
## Description 
This repository presents a fully automated pipeline for high-speed video microscopy (HSVM) analysis of respiratory cilia. The framework integrates: YOLOv8-based cilia segmentation,Dense optical flow (TV-L1),Biomechanically interpretable motion feature extraction, (vorticity, strain, motion-mask overlap (MMO), angular stability, wave-based periodicity, and ciliary beat frequency), Hierarchical machine-learning models for motion phenotype classification. The system is designed for video-level analysis and classifies ciliary motion phenotypes (immotile, normal, stiff, circular) in the context of Primary Ciliary Dyskinesia (PCD). Feature extraction and frequency estimation are performed independently of clinical labels to avoid information leakage.
## Contents
- Code for segmentation, optical-flow analysis, feature extraction, and classification
-   Representative HSVM example videos for demonstration purposes
-  Scripts to reproduce the main experiments reported in the manuscript

---

## Data Availability

All analysis code and representative HSVM example videos are publicly available at https://github.com/elfsvr/Cilia-motion-phenotyping-.
The full clinical HSVM dataset is archived at Marmara University and cannot be publicly released due to patient privacy and ethical restrictions.
---


## Requirements

Python >= 3.9

Install dependencies manually if `requirements.txt` is not provided:

```bash
pip install numpy pandas scipy scikit-learn imbalanced-learn \
            matplotlib seaborn shap opencv-python ultralytics

### Minimal dependencies
bash
pip install numpy pandas scipy scikit-learn imbalanced-learn matplotlib seaborn shap opencv-python ultralytics

## Contact
For questions, please contact the first author.

##Code structure
Yolov8segmentation.py Automated cilia segmentation from HSVM videos. Frames are automatically extracted before segmentation. tvl1_optical_flow.py Dense optical flow computation restricted to segmented cilia regions. feature_bank.py Construction of a reproducible feature bank including motion magnitude statistics, angular features, motion-mask overlap (MMO), wave-based descriptors, and ciliary beat frequency (CBF). classification.py Random Forest-based hierarchical and multi-class classification of motion phenotypes. main.py End-to-end execution of the complete pipeline.
## Automated segmentation (YOLOv8)

This step performs automated cilia segmentation from input videos. Frames are automatically extracted from each video before segmentation.


python Yolov8segmentation.py \
  --input_videos data/videos \
  --weights models/yolov8_seg.pt \
  --out_dir outputs/segmentation \
  --num_frames 30
  outputs/segmentation/
├── frames/ 
├── masks/
  
## optical flow
python tvl1_optical_flow.py \
  --frames_root outputs/segmentation/frames \
  --masks_root outputs/segmentation/masks \
  --out_dir data/motion

  

## feature bank
python feature_bank.py \
  --motion_dir data/motion \
  --out_csv outputs/features/feature_bank.csv

##  RF classification
python classification.py \
  --feature_csv outputs/features/feature_bank.csv \
  --out_dir outputs/figures

## MAIN
python main.py \
  --videos data/videos \
  --out_root outputs

##### Methodological Notes
- Feature extraction is performed at the video level.
-  Model evaluation is conducted at the patient level using group-aware cross-validation to prevent data leakage.
-  Optical-flow–derived motion signals are used for biomechanically interpretable feature extraction.
-  Ciliary beat frequency (CBF) is estimated using label-free signal-processing methods (autocorrelation and spectral analysis).
- Class imbalance is handled using resampling strategies applied to the training data only.
-  Balanced accuracy is used as the primary metric for model selection due to class imbalance.

 ## License
This project is released under the MIT License. 


