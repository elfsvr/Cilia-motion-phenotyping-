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

## Dataset Information

High-speed video microscopy recordings (HSVM) used in this study originate from two sources:

1. **Retrospectively archived clinical HSVM recordings (Marmara University Pediatric Pulmonology Department).**  
   These data were collected under institutional approval and were fully anonymized prior to analysis. Due to patient privacy and ethical restrictions, the full clinical dataset cannot be publicly released. Multiple recordings from the same individual were grouped using unique subject identifiers to prevent sample inflation.

2. **Publicly available HSVM recordings**, obtained from previously published and accessible sources, including:

   - Lavie & Amirav (2019), American Journal of Respiratory and Critical Care Medicine,  
     doi:10.1164/rccm.201904-0773LE  

   - Bottier et al. (2017), PLoS Computational Biology,  
     doi:10.1371/journal.pcbi.1005605  

   - Sampaio et al. (2021), ERJ Open Research,  
     doi:10.1183/23120541.00792-2020  

   - Jackson and Bottier (2022), European Respiratory Journal,  
     doi:10.1183/13993003.02300-2021  

   - University of Münster HSVM repository (Department of Pediatric Pulmonology),  
     https://www.medizin.uni-muenster.de/en/pcd/research/high-speed-video-microscopy-analysis-hvma.html  

Representative example videos are provided in this repository for demonstration purposes.
---


## Usage Instructions

### 1. Install dependencies

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

##Code Information

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

##Full pipeline execution
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

Requirements

pip install numpy pandas scipy scikit-learn imbalanced-learn \ matplotlib seaborn shap opencv-contrib-python ultralytics
 ## License
This project is released under the MIT License. 

Citations

If you use this repository, please cite:

Sever E. Automated hierarchical classification of ciliary motion phenotypes using optical flow and machine-learning for primary ciliary dyskinesia. PeerJ Computer Science (under review).
