# TV-L1 OPTICAL FLOW
# Google Colab Compatible - TAMAMEN EKSİKSİZ KOD

import os
import json
import cv2
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import glob

# Matplotlib setup - Colab Compatible
import matplotlib
try:
    from google.colab import files
    matplotlib.use('Agg')
    IN_COLAB = True
    print("Running in Google Colab - using Agg backend")
    # Enable inline plotting in Colab
    try:
        from IPython import get_ipython
        ipython = get_ipython()
        if ipython is not None:
            ipython.run_line_magic('matplotlib', 'inline')
    except:
        pass
except ImportError:
    try:
        matplotlib.use('TkAgg')
        IN_COLAB = False
        print("Running locally - using TkAgg backend")
    except ImportError:
        matplotlib.use('Agg')
        IN_COLAB = False
        print("Using fallback Agg backend")

plt.rcParams['figure.figsize'] = (12, 8)

# Patient directories - TÜM HASTALAR (86 adet)
PATIENT_DIRS = {frames_path
}

# Analysis Parameters - ORJİNAL AYARLAR
JSON_PATH = "_annotations.coco.json"
THRESHOLD_PERCENTILE = 96
FRAME_SKIP = 1
MAX_FRAMES_PER_PATIENT = 30
RESULTS_OUTPUT_DIR = "cilia_analysis_results"

# Visualization Settings - ORJİNAL AYARLAR
SHOW_PLOTS = True
SAVE_PLOTS = True
MAX_VISUALIZATIONS_PER_PATIENT = 10

# TV-L1 Parameters - ORJİNAL PARAMETRELER
TV_L1_PARAMS = {
    'tau': 0.25,
    'lambda': 0.15,
    'theta': 0.3,
    'nscales': 5,
    'warps': 5,
    'epsilon': 0.01
}

def check_environment():
    """Check if environment is properly set up"""
    print("Environment Check:")
    print("-" * 30)

    try:
        import numpy
        print(f"✓ NumPy version: {numpy.__version__}")
    except ImportError as e:
        print(f"✗ NumPy import failed: {e}")
        return False

    try:
        import cv2
        # Try different ways to get OpenCV version
        try:
            version = cv2.__version__
            print(f"✓ OpenCV version: {version}")
        except AttributeError:
            try:
                version = getattr(cv2, 'cv2').__version__ if hasattr(cv2, 'cv2') else "Unknown"
                print(f"✓ OpenCV imported (version: {version})")
            except:
                print("✓ OpenCV imported (version unknown)")

        # Test if optflow is available
        try:
            flow = cv2.optflow.createOptFlow_DualTVL1()
            print("✓ TV-L1 optical flow available")
            del flow  # Clean up
        except AttributeError:
            print("✗ TV-L1 optical flow not available - need opencv-contrib-python")
            return False
        except Exception as e:
            print(f"✗ TV-L1 optical flow error: {e}")
            return False

    except ImportError as e:
        print(f"✗ OpenCV import failed: {e}")
        return False

    try:
        import matplotlib
        print(f"✓ Matplotlib version: {matplotlib.__version__}")
        print(f"✓ Backend: {matplotlib.get_backend()}")
    except ImportError as e:
        print(f"✗ Matplotlib import failed: {e}")
        return False

    print("✓ Environment check passed!")
    return True

def extract_frame_id(filename):
    """Extract frame ID from filename"""
    filename = os.path.splitext(filename)[0].lower()
    filename = re.sub(r'\.rf\.[a-f0-9]+', '', filename)
    filename = re.sub(r'(_png_jpg|_jpg|_png)$', '', filename)

    patterns = [
        r'frame(\d{4})', r'(\d{4})$', r'-(\d{4})[_-]', r'_(\d{4})[_-]',
        r'(\d{4})', r'(\d{3})$', r'frame(\d{3})', r'(\d{2})$'
    ]

    for pattern in patterns:
        match = re.search(pattern, filename)
        if match:
            return int(match.group(1))
    return None

def debug_frame_extraction(frame_folder):
    """Debug frame ID extraction"""
    print(f"\nDEBUGGING FRAME EXTRACTION")
    print("=" * 40)

    try:
        all_files = os.listdir(frame_folder)
        frame_files = []
        other_files = []

        for f in all_files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                if any(keyword in f.lower() for keyword in ['plot', 'stats', 'summary', 'motion_']):
                    other_files.append(f)
                    continue
                frame_files.append(f)
            else:
                other_files.append(f)

        print(f"Frame files: {len(frame_files)}")
        if other_files:
            print(f"Excluded files: {other_files}")

        successful = 0
        for i, frame_file in enumerate(sorted(frame_files)):
            frame_id = extract_frame_id(frame_file)
            if frame_id is not None:
                successful += 1
                if i < 5 or i >= len(frame_files) - 5:
                    print(f"  {i+1:2d}. {frame_file} -> ID {frame_id}")

        if len(frame_files) > 10:
            print(f"  ... (showing first 5 and last 5)")

        print(f"Extraction summary: Successful: {successful}")
        return frame_files

    except Exception as e:
        print(f"Error reading folder: {e}")
        return []

def debug_json_content(json_path):
    """Debug JSON content"""
    print("JSON CONTENT DEBUG")
    print("=" * 50)

    try:
        with open(json_path) as f:
            data = json.load(f)

        print(f"JSON keys: {list(data.keys())}")

        if 'images' in data:
            images = data['images']
            print(f"Total images: {len(images)}")

        return data

    except Exception as e:
        print(f"Error reading JSON: {e}")
        return None

def improved_json_matching(json_path, frame_folder):
    """Improved JSON matching"""
    print(f"\nIMPROVED JSON MATCHING")
    print("=" * 50)

    data = debug_json_content(json_path)
    if not data or 'images' not in data:
        return create_synthetic_matches(frame_folder)

    folder_name = os.path.basename(frame_folder).lower()
    print(f"\nFrame folder: {folder_name}")

    if folder_name.startswith("frames_"):
        pattern = folder_name[7:]
    else:
        pattern = folder_name

    print(f"Search pattern: {pattern}")

    all_images = data['images']

    # Strategy 1: Exact pattern match
    exact_matches = []
    for img in all_images:
        filename = img.get('file_name', '').lower()
        if filename.startswith(pattern.lower() + '-'):
            exact_matches.append(img)

    print(f"\nExact pattern match: {len(exact_matches)} matches")
    if exact_matches:
        return process_json_matches(exact_matches, data, frame_folder)

    # Strategy 2: Pattern variations
    pattern_matches = []
    patterns_to_try = [pattern, pattern.replace('_', '-'), pattern.replace('-', '_')]

    for test_pattern in patterns_to_try:
        for img in all_images:
            filename = img.get('file_name', '').lower()
            if test_pattern.lower() in filename and filename not in [m.get('file_name', '') for m in pattern_matches]:
                pattern_matches.append(img)

    print(f"Pattern variations match: {len(pattern_matches)} matches")
    if pattern_matches and len(pattern_matches) < 100:
        return process_json_matches(pattern_matches, data, frame_folder)

    print("\nNo reliable JSON matches found - using synthetic")
    return create_synthetic_matches(frame_folder)

def process_json_matches(json_files, full_data, frame_folder):
    """Process the matched JSON files"""

    frame_files = debug_frame_extraction(frame_folder)
    print(f"Frame files found: {len(frame_files)}")

    # Create ID mappings
    json_by_id = {}
    for img in json_files:
        frame_id = extract_frame_id(img['file_name'])
        if frame_id is not None:
            json_by_id[frame_id] = img

    frame_by_id = {}
    for frame_file in frame_files:
        frame_id = extract_frame_id(frame_file)
        if frame_id is not None:
            frame_by_id[frame_id] = frame_file

    print(f"JSON IDs extracted: {len(json_by_id)}")
    print(f"Frame IDs extracted: {len(frame_by_id)}")

    # Find common IDs
    common_ids = set(json_by_id.keys()) & set(frame_by_id.keys())

    print(f"Common frame IDs: {len(common_ids)}")

    if len(common_ids) == 0:
        return create_synthetic_matches(frame_folder)

    # Create final matches
    matches = []
    for frame_id in sorted(common_ids):
        img = json_by_id[frame_id]
        frame_file = frame_by_id[frame_id]
        frame_path = os.path.join(frame_folder, frame_file)

        # Get annotations
        segments = []
        for ann in full_data.get('annotations', []):
            if ann['image_id'] == img['id'] and ann.get('segmentation'):
                for seg in ann['segmentation']:
                    if isinstance(seg, list) and len(seg) >= 6:
                        segments.append(seg)

        matches.append({
            'json_file': img['file_name'],
            'frame_file': frame_file,
            'frame_path': frame_path,
            'width': img['width'],
            'height': img['height'],
            'segments': segments,
            'frame_id': frame_id
        })

    print(f"Final matches created: {len(matches)}")
    return matches

def create_synthetic_matches(frame_folder):
    """Create synthetic matches when JSON data unavailable"""
    print("Creating synthetic matches...")

    if not os.path.exists(frame_folder):
        return []

    try:
        all_files = os.listdir(frame_folder)
        frame_files = []
        for f in all_files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                if any(keyword in f.lower() for keyword in ['plot', 'stats', 'summary', 'motion_', 'visualization']):
                    continue
                frame_files.append(f)

    except Exception as e:
        print(f"Error reading folder: {e}")
        return []

    matches = []
    for frame_file in frame_files:
        frame_id = extract_frame_id(frame_file)
        if frame_id is not None:
            frame_path = os.path.join(frame_folder, frame_file)

            try:
                img = cv2.imread(frame_path)
                if img is not None:
                    height, width = img.shape[:2]
                    matches.append({
                        'json_file': f'synthetic_{frame_file}',
                        'frame_file': frame_file,
                        'frame_path': frame_path,
                        'width': width,
                        'height': height,
                        'segments': [],
                        'frame_id': frame_id
                    })
            except Exception as e:
                print(f"Error loading {frame_file}: {e}")

    matches = sorted(matches, key=lambda x: x['frame_id'])
    print(f"Created {len(matches)} synthetic matches")
    return matches

def load_and_match_data(json_path, frame_folder):
    """Main matching function"""
    return improved_json_matching(json_path, frame_folder)

def create_validated_tv_l1_flow():
    """Create TV-L1 optical flow with error handling"""
    try:
        optical_flow = cv2.optflow.createOptFlow_DualTVL1()

        if hasattr(optical_flow, 'setTau'):
            optical_flow.setTau(TV_L1_PARAMS['tau'])
        if hasattr(optical_flow, 'setLambda'):
            optical_flow.setLambda(TV_L1_PARAMS['lambda'])
        if hasattr(optical_flow, 'setTheta'):
            optical_flow.setTheta(TV_L1_PARAMS['theta'])
        if hasattr(optical_flow, 'setNscales'):
            optical_flow.setNscales(TV_L1_PARAMS['nscales'])
        elif hasattr(optical_flow, 'setScalesNumber'):
            optical_flow.setScalesNumber(TV_L1_PARAMS['nscales'])
        if hasattr(optical_flow, 'setWarps'):
            optical_flow.setWarps(TV_L1_PARAMS['warps'])
        elif hasattr(optical_flow, 'setWarpingsNumber'):
            optical_flow.setWarpingsNumber(TV_L1_PARAMS['warps'])
        if hasattr(optical_flow, 'setEpsilon'):
            optical_flow.setEpsilon(TV_L1_PARAMS['epsilon'])

        print("✓ TV-L1 optical flow created successfully")
        return optical_flow

    except AttributeError as e:
        print(f"✗ TV-L1 optical flow error: {e}")
        print("Solution: Install opencv-contrib-python")
        raise RuntimeError("TV-L1 optical flow not available. Run: !pip install opencv-contrib-python")
    except Exception as e:
        print(f"✗ Unexpected error creating optical flow: {e}")
        raise

def create_combined_mask(segments, height, width):
    """Create combined mask from segmentation polygons"""
    mask = np.zeros((height, width), dtype=np.uint8)
    for seg in segments:
        polygon = np.array(seg, dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask, [polygon], 1)
    return mask

def enhanced_preprocessing(img):
    """Enhanced preprocessing for cilia motion"""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img)

def calculate_comprehensive_flow_metrics(img1, img2, gt_mask, optical_flow):
    """Calculate comprehensive optical flow metrics"""

    img1_proc = enhanced_preprocessing(img1)
    img2_proc = enhanced_preprocessing(img2)

    flow = optical_flow.calc(img1_proc, img2_proc, None)
    u, v = flow[..., 0], flow[..., 1]
    mag, ang = cv2.cartToPolar(u, v)

    mag_filtered = cv2.medianBlur(mag.astype(np.float32), 5)

    if mag_filtered[mag_filtered > 0].size > 0:
        threshold = np.percentile(mag_filtered[mag_filtered > 0], THRESHOLD_PERCENTILE)
        motion_mask = (mag_filtered > threshold).astype(np.uint8)
    else:
        motion_mask = np.zeros_like(mag_filtered, dtype=np.uint8)

    motion_in_gt = motion_mask * gt_mask
    union = np.logical_or(gt_mask, motion_mask)
    iou = np.sum(motion_in_gt) / np.sum(union) if np.sum(union) > 0 else 0

    gt_region = gt_mask == 1

    if np.any(gt_region):
        mean_mag = np.mean(mag[gt_region])
        mean_ang = np.degrees(np.mean(ang[gt_region]))

        du_dy = np.gradient(u, axis=0)
        dv_dx = np.gradient(v, axis=1)
        vorticity = dv_dx - du_dy
        mean_vorticity = np.mean(vorticity[gt_region])

        du_dx = np.gradient(u, axis=1)
        dv_dy = np.gradient(v, axis=0)
        strain = du_dx + dv_dy
        mean_strain = np.mean(strain[gt_region])

        dominant_angle = np.degrees(np.median(ang[gt_region]))
    else:
        mean_mag = mean_ang = mean_vorticity = mean_strain = 0
        dominant_angle = 0

    motion_coverage = np.sum(motion_in_gt) / np.sum(gt_mask) if np.sum(gt_mask) > 0 else 0

    return {
        'iou': iou,
        'motion_coverage': motion_coverage,
        'mean_magnitude': mean_mag,
        'mean_angle_deg': mean_ang,
        'vorticity': mean_vorticity,
        'strain': mean_strain,
        'dominant_angle_deg': dominant_angle,
        'flow_data': (u, v, mag, ang, vorticity, strain),
        'masks': (gt_mask, motion_mask, motion_in_gt)
    }

def create_enhanced_visualization(img1, img2, metrics, frame_info, save_path=None):
    """Create enhanced visualization"""

    print(f"Creating enhanced visualization for: {frame_info['title']}")

    u, v, mag, ang, vorticity, strain = metrics['flow_data']
    gt_mask, motion_mask, motion_in_gt = metrics['masks']

    fig = plt.figure(figsize=(20, 12))
    fig.suptitle(f'Enhanced Cilia Motion Analysis: {frame_info["title"]}', fontsize=16, fontweight='bold')

    gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)

    # Row 1: Original images and masks
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(img1, cmap='gray')
    ax1.set_title('Original Frame 1', fontsize=12)
    ax1.axis('off')

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(img2, cmap='gray')
    ax2.set_title('Original Frame 2', fontsize=12)
    ax2.axis('off')

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(gt_mask, cmap='gray')
    ax3.set_title('Cilia Ground Truth', fontsize=12)
    ax3.axis('off')

    ax4 = fig.add_subplot(gs[0, 3])
    ax4.imshow(motion_mask, cmap='gray')
    ax4.set_title(f'Detected Motion\n(Threshold: {THRESHOLD_PERCENTILE/100:.1f})', fontsize=12)
    ax4.axis('off')

    # Row 2: Flow analysis
    ax5 = fig.add_subplot(gs[1, 0])
    im1 = ax5.imshow(mag, cmap='viridis')
    ax5.set_title(f'Flow Magnitude\nMean: {metrics["mean_magnitude"]:.3f}px', fontsize=12)
    ax5.axis('off')
    plt.colorbar(im1, ax=ax5, fraction=0.046)

    ax6 = fig.add_subplot(gs[1, 1])
    flow_hsv = np.zeros((mag.shape[0], mag.shape[1], 3), dtype=np.uint8)
    flow_hsv[:,:,0] = (ang * 180 / np.pi / 2).astype(np.uint8)
    flow_hsv[:,:,1] = 255
    flow_hsv[:,:,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    flow_rgb = cv2.cvtColor(flow_hsv, cv2.COLOR_HSV2RGB)
    ax6.imshow(flow_rgb)
    ax6.set_title(f'Flow Direction\nDominant: {metrics["dominant_angle_deg"]:.1f}°', fontsize=12)
    ax6.axis('off')

    ax7 = fig.add_subplot(gs[1, 2])
    ax7.imshow(motion_in_gt, cmap='hot')
    ax7.set_title(f'Motion Overlap\nIoU: {metrics["iou"]:.3f}', fontsize=12)
    ax7.axis('off')

    # Coverage pie chart
    ax8 = fig.add_subplot(gs[1, 3])
    coverage = metrics['motion_coverage']
    precision = np.sum(motion_in_gt) / np.sum(motion_mask) if np.sum(motion_mask) > 0 else 0

    detected = coverage * 100
    missed = (1 - coverage) * 100

    sizes = [detected, missed]
    labels = [f'Detected\n{detected:.1f}%', f'Missed\n{missed:.1f}%']
    colors = ['#1f77b4', '#ff7f0e']

    ax8.pie(sizes, labels=labels, colors=colors, autopct='', startangle=90)
    ax8.set_title(f'Coverage: {coverage:.3f}\nPrecision: {precision:.3f}', fontsize=12)

    # Row 3: Advanced metrics
    ax9 = fig.add_subplot(gs[2, 0])
    im2 = ax9.imshow(vorticity, cmap='RdBu_r')
    ax9.set_title(f'Vorticity (Rotation)\n{metrics["vorticity"]:.4f}', fontsize=12)
    ax9.axis('off')
    plt.colorbar(im2, ax=ax9, fraction=0.046)

    ax10 = fig.add_subplot(gs[2, 1])
    im3 = ax10.imshow(strain, cmap='RdYlBu_r')
    ax10.set_title(f'Strain Rate\n{metrics["strain"]:.4f}', fontsize=12)
    ax10.axis('off')
    plt.colorbar(im3, ax=ax10, fraction=0.046)

    # Flow magnitude distribution
    ax11 = fig.add_subplot(gs[2, 2])
    mag_flat = mag.flatten()
    mag_nonzero = mag_flat[mag_flat > 0]

    if len(mag_nonzero) > 0:
        ax11.hist(mag_nonzero, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        median_mag = np.median(mag_nonzero)
        percentile_95 = np.percentile(mag_nonzero, 95)
        ax11.axvline(median_mag, color='red', linestyle='--', label=f'Median: {median_mag:.3f}')
        ax11.axvline(percentile_95, color='orange', linestyle='--', label=f'95th: {percentile_95:.3f}')
        ax11.legend()

    ax11.set_title('Flow Magnitude Distribution', fontsize=12)
    ax11.set_xlabel('Magnitude (pixels)')
    ax11.set_ylabel('Frequency')

    # Analysis Summary
    ax12 = fig.add_subplot(gs[2, 3])
    ax12.axis('off')

    summary_text = f"""MOTION ANALYSIS SUMMARY

IoU (Overlap): {metrics['iou']:.3f}
Coverage: {coverage:.3f}
Precision: {precision:.3f}
Coherence: {metrics['iou']:.3f}
Noise Ratio: {1-precision:.3f}

FLOW CHARACTERISTICS:
Mean Magnitude: {metrics['mean_magnitude']:.3f}px
Std Magnitude: {np.std(mag_nonzero) if len(mag_nonzero) > 0 else 0:.3f}px
Dominant Angle: {metrics['dominant_angle_deg']:.1f}°

ADVANCED METRICS:
Vorticity: {metrics['vorticity']:.4f}
Strain Rate: {metrics['strain']:.4f}
Complexity: {np.std(ang.flatten()):.3f}

TV-L1 PARAMETERS:
τ = {TV_L1_PARAMS['tau']}, λ = {TV_L1_PARAMS['lambda']}, θ = {TV_L1_PARAMS['theta']}
Scales = {TV_L1_PARAMS['nscales']}, Warps = {TV_L1_PARAMS['warps']}"""

    ax12.text(0.05, 0.95, summary_text, transform=ax12.transAxes, fontsize=10,
              verticalalignment='top', fontfamily='monospace',
              bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))

    # Row 4: Vector field and motion profile
    ax13 = fig.add_subplot(gs[3, :2])

    step = max(1, min(mag.shape) // 20)
    Y, X = np.mgrid[0:mag.shape[0]:step, 0:mag.shape[1]:step]
    U = u[::step, ::step]
    V = v[::step, ::step]
    M = mag[::step, ::step]

    ax13.imshow(img1, cmap='gray', alpha=0.7)
    quiver = ax13.quiver(X, Y, U, V, M, cmap='hot', scale=50, alpha=0.8)
    ax13.set_title('Motion Vector Field Overlay', fontsize=12)
    ax13.axis('off')
    plt.colorbar(quiver, ax=ax13, fraction=0.046)

    ax14 = fig.add_subplot(gs[3, 2:])
    motion_profile = np.sum(motion_in_gt, axis=0) if motion_in_gt.ndim > 1 else motion_in_gt
    if len(motion_profile) > 1:
        ax14.plot(motion_profile, color='blue', linewidth=2)
        ax14.fill_between(range(len(motion_profile)), motion_profile, alpha=0.3, color='blue')
        ax14.set_title('Horizontal Motion Profile', fontsize=12)
        ax14.set_xlabel('X Position (pixels)')
        ax14.set_ylabel('Motion Intensity')
        ax14.grid(True, alpha=0.3)
    else:
        ax14.text(0.5, 0.5, 'Motion Profile\nNot Available',
                  ha='center', va='center', transform=ax14.transAxes)
        ax14.set_title('Motion Profile', fontsize=12)

    plt.tight_layout()

    if save_path and SAVE_PLOTS:
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved enhanced visualization: {os.path.basename(save_path)}")
        except Exception as e:
            print(f"Failed to save: {e}")

    if SHOW_PLOTS:
        if IN_COLAB:
            plt.show()
            print("Enhanced plot displayed in Colab")
        else:
            plt.show()
            print("Enhanced plot displayed locally")

    return fig

def extract_patient_name(patient_dir):
    """Extract patient name from directory path"""
    dirname = os.path.basename(patient_dir)
    if dirname.startswith('frames_'):
        return dirname[7:]
    return dirname

def convert_existing_to_legacy():
    """Convert existing CSV files to legacy format"""
    print("\n" + "="*60)
    print("CONVERTING EXISTING CSV FILES TO LEGACY FORMAT")
    print("="*60)

    base_dir = RESULTS_OUTPUT_DIR
    converted_count = 0

    if not os.path.exists(base_dir):
        print(f"Results directory does not exist: {base_dir}")
        return

    for folder in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder)

        if not os.path.isdir(folder_path) or folder == 'visualizations':
            continue

        csv_path = os.path.join(folder_path, f"{folder}_motion_stats.csv")

        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                print(f"Converting {folder}...")

                required_columns = ['frame_pair', 'iou', 'motion_coverage', 'num_annotations',
                                  'mean_magnitude', 'mean_angle_deg', 'mean_vorticity',
                                  'mean_strain', 'dominant_angle_deg']

                missing_columns = [col for col in required_columns if col not in df.columns]
                if missing_columns:
                    print(f"  Warning: Missing columns in {folder}: {missing_columns}")
                    continue

                legacy_df = pd.DataFrame({
                    'frame': df['frame_pair'],
                    'iou': df['iou'],
                    'motion_in_gt_ratio': df['motion_coverage'],
                    'num_annotations': df['num_annotations'],
                    'mean_magnitude': df['mean_magnitude'],
                    'mean_angle_deg': df['mean_angle_deg'],
                    'mean_vorticity': df['mean_vorticity'],
                    'mean_strain': df['mean_strain'],
                    'dominant_angle_deg': df['dominant_angle_deg']
                })

                legacy_path = os.path.join(folder_path, f"{folder}_motion_stats_30frames_interp.csv")
                legacy_df.to_csv(legacy_path, index=False)
                converted_count += 1
                print(f"  Created: {os.path.basename(legacy_path)} ({len(legacy_df)} rows)")

            except Exception as e:
                print(f"  Error converting {folder}: {e}")
        else:
            print(f"  No CSV found for {folder}")

    print(f"\nConversion complete: {converted_count} legacy CSV files created")
    print("="*60)

def process_patient_directory(patient_dir, optical_flow):
    """Process a single patient directory"""
    patient_name = extract_patient_name(patient_dir)

    print(f"\nProcessing: {patient_name}")
    print(f"Directory: {patient_dir}")

    if not os.path.exists(patient_dir):
        print(f"Directory not found: {patient_dir}")
        return []

    matched_data = load_and_match_data(JSON_PATH, patient_dir)

    if not matched_data:
        print("No matched data found!")
        return []

    matched_data = matched_data[:MAX_FRAMES_PER_PATIENT]
    results = []
    processed_pairs = 0

    print(f"Processing {len(matched_data) - FRAME_SKIP} frame pairs...")

    for i in range(len(matched_data) - FRAME_SKIP):
        current = matched_data[i]
        next_frame = matched_data[i + FRAME_SKIP]

        frame_gap = next_frame['frame_id'] - current['frame_id']

        img1 = cv2.imread(current['frame_path'], cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(next_frame['frame_path'], cv2.IMREAD_GRAYSCALE)

        if img1 is None or img2 is None:
            print(f"Could not load images for pair {i+1}")
            continue

        if current['segments']:
            gt_mask = create_combined_mask(current['segments'], current['height'], current['width'])
            if gt_mask.shape != img1.shape:
                gt_mask = cv2.resize(gt_mask, (img1.shape[1], img1.shape[0]), interpolation=cv2.INTER_NEAREST)
            num_annotations = len(current['segments'])
            print(f"   Frame {current['frame_id']}: {num_annotations} annotations found")
        else:
            mask = np.zeros((img1.shape[0], img1.shape[1]), dtype=np.uint8)
            center_y, center_x = img1.shape[0] // 2, img1.shape[1] // 2
            region_h, region_w = img1.shape[0] // 3, img1.shape[1] // 3
            y1 = max(0, center_y - region_h // 2)
            y2 = min(img1.shape[0], center_y + region_h // 2)
            x1 = max(0, center_x - region_w // 2)
            x2 = min(img1.shape[1], center_x + region_w // 2)
            mask[y1:y2, x1:x2] = 1
            gt_mask = mask
            num_annotations = 1
            print(f"   Frame {current['frame_id']}: Using synthetic mask (no annotations)")

        print(f"Calculating metrics for pair {processed_pairs + 1}...")
        metrics = calculate_comprehensive_flow_metrics(img1, img2, gt_mask, optical_flow)

        frame_name = f"frame{current['frame_id']:04d}_{next_frame['frame_id']:04d}"

        result = {
            'patient_name': patient_name,
            'frame_pair': frame_name,
            'frame_1_id': current['frame_id'],
            'frame_2_id': next_frame['frame_id'],
            'frame_gap': frame_gap,
            'iou': round(metrics['iou'], 4),
            'motion_coverage': round(metrics['motion_coverage'], 4),
            'num_annotations': num_annotations,
            'mean_magnitude': round(metrics['mean_magnitude'], 4),
            'mean_angle_deg': round(metrics['mean_angle_deg'], 4),
            'mean_vorticity': round(metrics['vorticity'], 6),
            'mean_strain': round(metrics['strain'], 6),
            'dominant_angle_deg': round(metrics['dominant_angle_deg'], 4),
            'has_real_annotations': len(current['segments']) > 0,
            'image_width': current['width'],
            'image_height': current['height']
        }

        results.append(result)
        processed_pairs += 1

        if processed_pairs <= MAX_VISUALIZATIONS_PER_PATIENT:
            print(f"\nCreating visualization {processed_pairs}...")

            frame_info = {
                'title': f'{patient_name} | Frame {current["frame_id"]} → {next_frame["frame_id"]}',
                'gap': frame_gap
            }

            save_path = None
            if SAVE_PLOTS:
                safe_patient_name = re.sub(r'[^\w\-_.]', '_', patient_name)
                viz_folder = os.path.join(RESULTS_OUTPUT_DIR, "visualizations", safe_patient_name)
                save_path = os.path.join(viz_folder, f"motion_{current['frame_id']:04d}_{next_frame['frame_id']:04d}.png")

            try:
                fig = create_enhanced_visualization(img1, img2, metrics, frame_info, save_path)
                print(f"Enhanced visualization {processed_pairs} completed successfully")

                if SHOW_PLOTS and not IN_COLAB:
                    input("Press Enter to continue...")
                elif SHOW_PLOTS and IN_COLAB:
                    print("Next visualization coming up...")
                    import time
                    time.sleep(2)

            except Exception as e:
                print(f"Enhanced visualization {processed_pairs} error: {e}")
                import traceback
                traceback.print_exc()

        print(f"Pair {processed_pairs}: IoU={metrics['iou']:.3f}, Coverage={metrics['motion_coverage']:.3f}")

    print(f"Completed processing {patient_name}: {len(results)} frame pairs")
    return results

def main():
    """Main function for cilia motion analysis"""

    print("Multi-Patient Cilia Motion Analysis")
    print("=" * 60)

    # Check environment first
    if not check_environment():
        print("Environment check failed. Please fix dependencies first.")
        return

    print(f"Processing {len(PATIENT_DIRS)} patients")
    print(f"Max frames per patient: {MAX_FRAMES_PER_PATIENT}")
    print(f"Output directory: {RESULTS_OUTPUT_DIR}")
    print(f"Matplotlib backend: {matplotlib.get_backend()}")
    print(f"Show plots: {SHOW_PLOTS}")
    print(f"Save plots: {SAVE_PLOTS}")
    print(f"Running in Colab: {IN_COLAB}")

    if IN_COLAB:
        print("Colab detected - plots will be displayed inline")
    else:
        print("Local environment detected")
        if matplotlib.get_backend() == 'Agg':
            print("Warning: Non-interactive backend detected.")

    print("=" * 60)

    os.makedirs(RESULTS_OUTPUT_DIR, exist_ok=True)

    # Convert existing files to legacy format first
    convert_existing_to_legacy()

    optical_flow = create_validated_tv_l1_flow()

    all_results = []
    processed_patients = 0

    patient_dirs_to_process = list(sorted(PATIENT_DIRS))
    print(f"PROCESSING ALL {len(patient_dirs_to_process)} PATIENTS")

    for patient_idx, patient_dir in enumerate(patient_dirs_to_process, 1):
        print(f"\nPATIENT {patient_idx}/{len(patient_dirs_to_process)}: {os.path.basename(patient_dir)}")
        print("-" * 80)
        try:
            print(f"DEBUG: Starting to process patient directory: {patient_dir}")
            patient_results = process_patient_directory(patient_dir, optical_flow)
            print(f"DEBUG: Patient results length: {len(patient_results) if patient_results else 0}")

            if patient_results:
                print("DEBUG: Patient results found, proceeding to save files...")
                all_results.extend(patient_results)
                processed_patients += 1

                patient_name = extract_patient_name(patient_dir)
                safe_patient_name = re.sub(r'[^\w\-_.]', '_', patient_name)

                patient_results_dir = os.path.join(RESULTS_OUTPUT_DIR, safe_patient_name)
                os.makedirs(patient_results_dir, exist_ok=True)

                # Save NEW FORMAT CSV
                csv_path = os.path.join(patient_results_dir, f"{safe_patient_name}_motion_stats.csv")
                df = pd.DataFrame(patient_results)
                df.to_csv(csv_path, index=False)
                print(f"SAVED NEW FORMAT CSV: {os.path.basename(csv_path)} ({len(patient_results)} rows)")

                # CREATE LEGACY FORMAT CSV
                print("*** CREATING LEGACY FORMAT CSV ***")
                legacy_results = []
                for result in patient_results:
                    legacy_result = {
                        'frame': result['frame_pair'],
                        'iou': result['iou'],
                        'motion_in_gt_ratio': result['motion_coverage'],
                        'num_annotations': result['num_annotations'],
                        'mean_magnitude': result['mean_magnitude'],
                        'mean_angle_deg': result['mean_angle_deg'],
                        'mean_vorticity': result['mean_vorticity'],
                        'mean_strain': result['mean_strain'],
                        'dominant_angle_deg': result['dominant_angle_deg']
                    }
                    legacy_results.append(legacy_result)

                legacy_csv_path = os.path.join(patient_results_dir, f"{safe_patient_name}_motion_stats_30frames_interp.csv")

                try:
                    legacy_df = pd.DataFrame(legacy_results)
                    legacy_df.to_csv(legacy_csv_path, index=False)
                    print(f"*** SAVED LEGACY FORMAT CSV: {os.path.basename(legacy_csv_path)} ({len(legacy_results)} rows) ***")
                except Exception as e:
                    print(f"*** ERROR creating legacy CSV: {e} ***")

                # Save summary
                summary_stats = {
                    'patient_name': patient_name,
                    'total_frame_pairs': len(patient_results),
                    'frames_with_annotations': sum(1 for r in patient_results if r['has_real_annotations']),
                    'avg_iou': float(np.mean([r['iou'] for r in patient_results])),
                    'avg_motion_coverage': float(np.mean([r['motion_coverage'] for r in patient_results])),
                    'avg_magnitude': float(np.mean([r['mean_magnitude'] for r in patient_results])),
                    'avg_vorticity': float(np.mean([r['mean_vorticity'] for r in patient_results])),
                    'avg_strain': float(np.mean([r['mean_strain'] for r in patient_results]))
                }

                summary_path = os.path.join(patient_results_dir, f"{safe_patient_name}_summary.json")
                with open(summary_path, 'w') as f:
                    json.dump(summary_stats, f, indent=2)
                print(f"Saved Summary: {os.path.basename(summary_path)}")

                print(f"Patient {patient_idx} completed: {len(patient_results)} frame pairs")
            else:
                print(f"No results for patient {patient_idx}")

        except Exception as e:
            print(f"Error processing patient {patient_idx} ({patient_dir}): {e}")
            import traceback
            traceback.print_exc()
            print("Continuing to next patient...\n")

    # Save combined results
    if all_results:
        combined_df = pd.DataFrame(all_results)
        combined_path = os.path.join(RESULTS_OUTPUT_DIR, "all_patients_motion_stats.csv")
        combined_df.to_csv(combined_path, index=False)

        numeric_columns = ['iou', 'motion_coverage', 'mean_magnitude', 'mean_vorticity', 'mean_strain']

        print(f"\nANALYSIS COMPLETE!")
        print(f"Processed patients: {processed_patients}")
        print(f"Total frame pairs: {len(all_results)}")
        print(f"Results saved to: {RESULTS_OUTPUT_DIR}")

        print(f"\nSUMMARY STATISTICS:")
        for col in numeric_columns:
            if col in combined_df.columns:
                mean_val = combined_df[col].mean()
                std_val = combined_df[col].std()
                min_val = combined_df[col].min()
                max_val = combined_df[col].max()
                print(f"   {col:20s}: {mean_val:8.4f} ± {std_val:8.4f} (min: {min_val:8.4f}, max: {max_val:8.4f})")

        print("="*60)
    else:
        print("No results generated")

# MAIN EXECUTION
if __name__ == "__main__":
    print("="*60)
    print("ENHANCED CILIA MOTION ANALYSIS - STARTING")
    print("="*60)

    print("Initial matplotlib setup:")
    print(f"Backend: {matplotlib.get_backend()}")
    print(f"IN_COLAB: {IN_COLAB}")

    if IN_COLAB:
        try:
            from IPython import get_ipython
            ipython = get_ipython()
            if ipython is not None:
                ipython.magic('matplotlib inline')
                print("Set matplotlib inline for Colab")
        except Exception as e:
            print(f"Could not set matplotlib inline: {e}")
    else:
        try:
            plt.ion()
            print("Interactive plotting enabled")
        except Exception as e:
            print(f"Could not enable interactive plotting: {e}")

    print("="*60)

    try:
        main()

        print("\n" + "="*60)
        print("ANALYSIS COMPLETED SUCCESSFULLY!")
        print("="*60)

    except KeyboardInterrupt:
        print("\n" + "="*60)
        print("ANALYSIS INTERRUPTED BY USER")
        print("="*60)

    except Exception as e:
        print("\n" + "="*60)
        print("ANALYSIS FAILED WITH ERROR:")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        print("="*60)

    finally:
        print("\nScript execution finished.")
        print("Check the results directory for output files.")