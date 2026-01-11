import argparse
import os
import cv2
from ultralytics import YOLO # type: ignore
import numpy as np

VIDEO_EXTENSIONS = [".mp4", ".avi", ".mov", ".mkv"]

def extract_frames_from_video(video_path, output_dir, num_frames):
    """Extract evenly spaced frames from a video file."""
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Get total number of frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        print(f"Error: No frames found in video {video_path}")
        cap.release()
        return
    
    # Calculate frame indices to extract (evenly spaced)
    frame_indices = []
    if total_frames <= num_frames:
        # If video has fewer frames than requested, take all frames
        frame_indices = list(range(total_frames))
    else:
        # Calculate evenly spaced frame indices
        step = total_frames // (num_frames + 1)
        frame_indices = [step * (i + 1) for i in range(num_frames)]
    
    # Get the video filename with extension for naming output files
    video_filename = os.path.basename(video_path)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract frames
    for i, frame_index in enumerate(frame_indices, start=1):
        # Set video position to the desired frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        
        # Read the frame
        ret, frame = cap.read()
        
        if ret:
            # Create output filename: "video_name.extension frame N"
            output_filename = f"{video_filename} frame {i}.jpg"
            output_path = os.path.join(output_dir, output_filename)
            
            # Save the frame
            cv2.imwrite(output_path, frame)
            print(f"Saved: {output_filename}")
        else:
            print(f"Error: Could not read frame {frame_index} from {video_path}")
    
    # Release the video capture object
    cap.release()

def extract_frames_from_video_directory(videos_dir, output_directory, num_frames):
    """
    Extract frames from all videos in a directory.
    """
    # Walk through all subdirectories
    for root, dirs, files in os.walk(videos_dir):
        for file in files:
            # Check if file has a video extension
            file_extension = os.path.splitext(file)[1].lower()
            if file_extension in VIDEO_EXTENSIONS:
                # Get full video path
                video_path = os.path.join(root, file)
                
                # Get relative path from videos_dir to maintain directory structure
                relative_path = os.path.relpath(root, videos_dir)
                
                # Create corresponding output directory
                video_output_dir = os.path.join(output_directory, relative_path)
                
                print(f"\nProcessing: {os.path.join(relative_path, file)}")
                
                # Extract frames from this video
                extract_frames_from_video(video_path, video_output_dir, num_frames)
    
    print(f"\nCompleted! All frames saved to '{output_directory}' folder.")

def create_labels_for_frames(model, frames_dir, labels_dir):
    """
    Create COCO format labels for frames using YOLO model predictions.
    """
    import json
    import datetime
    
    os.makedirs(labels_dir, exist_ok=True)
    
    # Get all image files
    image_files = []
    for root, dirs, files in os.walk(frames_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                image_files.append(os.path.join(root, file))
    
    if not image_files:
        print(f"No image files found in {frames_dir}")
        return
    
    # Initialize COCO format structure
    coco_format = {
        "info": {
            "year": "2025",
            "version": "1.0",
            "description": "Cilia Detection YOLO Predictions",
            "date_created": datetime.datetime.now().isoformat()
        },
        "licenses": [{
            "id": 1,
            "url": "https://creativecommons.org/licenses/by/4.0/",
            "name": "CC BY 4.0"
        }],
        "categories": [{"id": 0, "name": "Cilia"}],
        "images": [],
        "annotations": []
    }
    
    ann_id = 1
    
    print(f"\nGenerating predictions for {len(image_files)} frames...")
    
    for idx, img_path in enumerate(image_files):
        # Read image to get dimensions
        img = cv2.imread(img_path)
        if img is None:
            print(f"Error: Could not read {img_path}")
            continue
        
        H, W = img.shape[:2]
        
        # Get relative path for file_name
        file_name = os.path.relpath(img_path, frames_dir)
        
        # Add image to COCO format
        coco_format["images"].append({
            "id": idx,
            "file_name": file_name,
            "width": W,
            "height": H
        })
        
        # Run YOLO prediction
        results = model.predict(img_path, imgsz=640, verbose=False)
        result = results[0]
        
        # Process masks if available
        if result.masks is not None:
            for mask in result.masks.xy:  # Polygon format
                poly = mask.reshape(-1).tolist()
                x_coords = poly[0::2]
                y_coords = poly[1::2]
                
                if len(x_coords) < 3 or len(y_coords) < 3:
                    continue
                
                x_min, y_min = min(x_coords), min(y_coords)
                x_max, y_max = max(x_coords), max(y_coords)
                
                coco_format["annotations"].append({
                    "id": ann_id,
                    "image_id": idx,
                    "category_id": 0,
                    "segmentation": [poly],
                    "area": float((x_max - x_min) * (y_max - y_min)),
                    "bbox": [x_min, y_min, x_max - x_min, y_max - y_min],
                    "iscrowd": 0
                })
                ann_id += 1
        
        if (idx + 1) % 10 == 0:
            print(f"Processed {idx + 1}/{len(image_files)} frames")
    
    # Save COCO JSON
    json_path = os.path.join(labels_dir, "coco_annotations.json")
    with open(json_path, "w") as f:
        json.dump(coco_format, f, indent=2)
    
    print(f"\n✅ COCO annotations saved to: {json_path}")
    print(f"Total annotations: {len(coco_format['annotations'])}")

def visualize_detections(model, frames_dir, predictions_dir, output_dir, labels_dir=None):
    """
    Visualize detections on frames using existing COCO annotations.
    Optionally overlay ground truth labels if labels_dir is provided.
    
    Args:
        model: YOLO model instance (not used, kept for consistency)
        frames_dir: Directory containing input frames
        predictions_dir: Directory containing COCO predictions JSON
        output_dir: Directory to save visualized images
        labels_dir: Optional directory containing ground truth YOLO labels
    """
    import json
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load COCO annotations
    json_path = os.path.join(predictions_dir, "coco_annotations.json")
    if not os.path.exists(json_path):
        print(f"Error: COCO annotations not found at {json_path}")
        return
    
    with open(json_path, 'r') as f:
        coco_data = json.load(f)
    
    print(f"\nGenerating visualizations from COCO annotations...")
    if labels_dir and os.path.exists(labels_dir):
        print(f"✓ Ground truth labels will be overlaid in green from: {labels_dir}")
    
    # Create image_id to annotations mapping
    img_id_to_anns = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in img_id_to_anns:
            img_id_to_anns[img_id] = []
        img_id_to_anns[img_id].append(ann)
    
    # Process each image
    for img_info in coco_data['images']:
        img_id = img_info['id']
        file_name = img_info['file_name']
        
        # Read image
        img_path = os.path.join(frames_dir, file_name)
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"Error: Could not read {img_path}")
            continue
        
        H, W = img.shape[:2]
        
        # --- GROUND TRUTH LABELS (GREEN POLYGONS) ---
        if labels_dir and os.path.exists(labels_dir):
            # Construct label file path (replace .jpg/.png with .txt)
            label_filename = os.path.splitext(file_name)[0] + '.txt'
            label_path = os.path.join(labels_dir, label_filename)
            
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) < 5:  # Need at least class + 2 points
                        continue
                    
                    cls = int(parts[0])
                    coords = list(map(float, parts[1:]))
                    
                    # Convert normalized coords to pixel coords
                    pts = np.array([[int(x*W), int(y*H)] for x, y in zip(coords[0::2], coords[1::2])], np.int32)
                    pts = pts.reshape((-1, 1, 2))
                    
                    # Draw green polygon
                    cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
        
        # --- MODEL PREDICTIONS (RED MASK) ---
        # Get annotations for this image
        anns = img_id_to_anns.get(img_id, [])
        
        # Create combined mask from all annotations
        combined_mask = np.zeros((H, W), dtype=np.uint8)
        
        for ann in anns:
            if 'segmentation' in ann and ann['segmentation']:
                poly = np.array(ann['segmentation'][0]).reshape(-1, 2)
                poly = poly.astype(np.int32)
                cv2.fillPoly(combined_mask, [poly], (255,))
        
        # Create red mask overlay
        colored_mask = cv2.merge([combined_mask, np.zeros_like(combined_mask), np.zeros_like(combined_mask)])
        img = cv2.addWeighted(img, 1, colored_mask, 0.5, 0)
        
        # Save visualization
        out_path = os.path.join(output_dir, file_name)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        cv2.imwrite(out_path, img)
        
        if (img_id + 1) % 10 == 0:
            print(f"Visualized {img_id + 1}/{len(coco_data['images'])} frames")
    
    print(f"\n✅ Visualizations saved to: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Cilia Detection")
    parser.add_argument("--videos_dir", type=str, default="videos-marmara", help="Directory of input videos")
    parser.add_argument("--num-frames", type=str, default=30, help="Number of frames to process")
    args = parser.parse_args()

    if not args.videos_dir:
        print("Please provide the videos directory using --videos_dir")
        return
    videos_directory = args.videos_dir

    if not args.num_frames:
        print("Please provide the number of frames to process using --num-frames")
        return
    num_frames = args.num_frames
    
    # * load model
    model = YOLO("cillia_detection_model.pt")

    # * extract frames
    frames_directory = f"frames-{videos_directory}"
    if not os.path.exists(frames_directory):
        os.makedirs(frames_directory)
        extract_frames_from_video_directory(videos_directory, frames_directory, int(num_frames))
        
    # * create labels for frames
    prediction_directory = f"predictions-{videos_directory}"
    if not os.path.exists(prediction_directory):
        os.makedirs(prediction_directory)
        create_labels_for_frames(model, frames_directory, prediction_directory)
    
    # * visualize
    visualization_output_directory = f"visualizations-{videos_directory}"
    if not os.path.exists(visualization_output_directory):
        os.makedirs(visualization_output_directory, exist_ok=True)
        if os.path.exists(f"labels-{videos_directory}"):
            labels_directory = f"labels-{videos_directory}"
        else:
            labels_directory = None
        visualize_detections(model, frames_directory, prediction_directory, visualization_output_directory, labels_directory)

    


if __name__ == "__main__":
    main()
