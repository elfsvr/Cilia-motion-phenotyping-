# Cilia-motion-phenotyping

Fully automated and interpretable framework for motion phenotype classification from high-speed microscopy videos.  
Provides reproducible feature extraction and hierarchical machine learning tools for biomedical video analysis.

---

The framework implements a fully automated pipeline for high-speed video microscopy (HSVM) analysis,  
including YOLO-based cilia segmentation, dense optical flow, biomechanically interpretable feature  
extraction (vorticity, strain, motion-mask overlap, angular stability, and wave-based periodicity),  
and a hierarchical machine learning architecture for classifying ciliary motion phenotypes  
(immotile, normal, stiff, circular) in the context of Primary Ciliary Dyskinesia (PCD).

---

## Contents
- Code for segmentation, optical-flow analysis, feature extraction, and classification
- Example HSVM videos for demonstration purposes
- Scripts to reproduce the main experiments reported in the manuscript

---

## Data Availability
The analysis code supporting the findings of this study, together with representative HSVM example videos,  
is publicly available at **[GitHub repository link]**.  

Due to patient privacy and ethical restrictions, the full clinical HSVM dataset cannot be publicly shared.  
Additional data may be made available upon reasonable request.

---

## Requirements

If `requirements.txt` is not provided, install the dependencies manually:

bash
pip install numpy pandas scipy scikit-learn imbalanced-learn matplotlib seaborn shap opencv-python ultralytics

### Minimal dependencies
bash
pip install numpy pandas scipy scikit-learn imbalanced-learn matplotlib seaborn shap opencv-python ultralytics

## Contact
For questions, please contact the first author.

## Automated segmentation (YOLOv8)

This step performs automated cilia segmentation from input videos.
Frames are automatically extracted from each video before segmentation.


python Yolov8segmentation.py \
  --input_videos data/videos \
  --weights models/yolov8_seg.pt \
  --out_dir outputs/segmentation \
  --num_frames 30
  outputs/segmentation/

  
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



  


