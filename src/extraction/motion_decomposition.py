import os
import re
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import skew, kurtosis
from ultralytics import YOLO, SAM
from tqdm import tqdm

if not hasattr(F, 'scaled_dot_product_attention') and hasattr(F, '_scaled_dot_product_attention'):
    print("[SYSTEM] Outdated PyTorch detected. Advanced Monkey-patching SDPA for SAM 2.1...")
    def _sdpa_wrapper(*args, **kwargs):
        # Run the old PyTorch 1.13 backend function
        out = F._scaled_dot_product_attention(*args, **kwargs)
        # If it returns a tuple (output, weights), strip the weights and return just the output
        return out[0] if isinstance(out, tuple) else out
    # Bind our wrapper to the standard PyTorch 2.0 namespace
    F.scaled_dot_product_attention = _sdpa_wrapper

TOOL_CLASSES = ["Grasper", "Bipolar", "Hook", "Scissors", "Clipper", "Irrigator", "SpecimenBag"]

# The 54 advanced statistical metrics
PHYSICS_METRICS = [
    "tool_jerk_mean", "tool_jerk_std", "tool_jerk_q05", "tool_jerk_q25", "tool_jerk_median", "tool_jerk_q75", "tool_jerk_q95", "tool_jerk_skew", "tool_jerk_kurt",
    "tool_div_mean", "tool_div_std", "tool_div_q05", "tool_div_q25", "tool_div_median", "tool_div_q75", "tool_div_q95", "tool_div_skew", "tool_div_kurt",
    "tool_curl_mean", "tool_curl_std", "tool_curl_q05", "tool_curl_q25", "tool_curl_median", "tool_curl_q75", "tool_curl_q95", "tool_curl_skew", "tool_curl_kurt",
    
    "env_jerk_mean", "env_jerk_std", "env_jerk_q05", "env_jerk_q25", "env_jerk_median", "env_jerk_q75", "env_jerk_q95", "env_jerk_skew", "env_jerk_kurt",
    "env_div_mean", "env_div_std", "env_div_q05", "env_div_q25", "env_div_median", "env_div_q75", "env_div_q95", "env_div_skew", "env_div_kurt",
    "env_curl_mean", "env_curl_std", "env_curl_q05", "env_curl_q25", "env_curl_median", "env_curl_q75", "env_curl_q95", "env_curl_skew", "env_curl_kurt"
]

# --- ADDED: 'tool_id' to the CSV columns ---
CSV_COLUMNS = ["video_name", "frame_idx", "tool_id", "tool_label", "phase_label"] + PHYSICS_METRICS

def calculate_spatial_kinematics(flow):
    u, v = flow[..., 0], flow[..., 1]
    uy, ux = np.gradient(u) 
    vy, vx = np.gradient(v)
    return ux + vy, vx - uy

def calculate_jerk(f_t, f_t1, f_t2):
    accel_t = f_t - f_t1
    accel_t1 = f_t1 - f_t2
    return np.linalg.norm((accel_t - accel_t1), axis=-1)

def extract_physics(mask, flow_map):
    """Extracts 9 advanced statistical metrics for the masked region."""
    valid_pixels = mask == 1
    if not np.any(valid_pixels):
        return [0.0] * 9
        
    data = flow_map[valid_pixels]
    
    mean_val = float(np.mean(data))
    std_val = float(np.std(data))
    q05 = float(np.percentile(data, 5))
    q25 = float(np.percentile(data, 25))
    median = float(np.median(data))
    q75 = float(np.percentile(data, 75))
    q95 = float(np.percentile(data, 95))
    
    if std_val < 1e-6:
        skew_val, kurt_val = 0.0, 0.0
    else:
        skew_val = float(skew(data, nan_policy='omit'))
        kurt_val = float(kurtosis(data, nan_policy='omit'))
        
    return [mean_val, std_val, q05, q25, median, q75, q95, skew_val, kurt_val]

def extract_frame_number(filepath):
    """Extracts the integer sequence number from a filename."""
    basename = os.path.basename(filepath)
    numbers = re.findall(r'\d+', basename)
    return int(numbers[-1]) if numbers else -1

def extract_video_physics(video_name, frames_dir, flow_dir, phase_labels_csv, yolo_model_path, output_csv_path):
    print(f"\n[INIT] Loading YOLO model...")
    yolo_model = YOLO(yolo_model_path)
    
    print(f"[INIT] Loading Ultralytics SAM 2.1 Base...")
    sam_predictor = SAM('sam2.1_b.pt')
    
    labels_df = pd.read_csv(phase_labels_csv, sep=r'\s+') if phase_labels_csv else None
    
    print("[INIT] Scanning and aligning dataset directories...")
    raw_frame_paths = [os.path.join(root, f) for root, _, files in os.walk(frames_dir) for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    raw_flow_paths = [os.path.join(root, f) for root, _, files in os.walk(flow_dir) for f in files if f.lower().endswith('.npy')]

    frame_dict = {extract_frame_number(p): p for p in raw_frame_paths}
    flow_dict = {extract_frame_number(p): p for p in raw_flow_paths}

    valid_keys = sorted(list(set(frame_dict.keys()) & set(flow_dict.keys())))

    if len(valid_keys) == 0:
        print("FATAL: Could not align any frames with flow files. Check your naming conventions.")
        return

    frame_paths = [frame_dict[k] for k in valid_keys]
    flow_paths = [flow_dict[k] for k in valid_keys]
    print(f"[SUCCESS] Aligned {len(valid_keys)} valid frame/flow pairs. Dropped {len(raw_frame_paths) - len(valid_keys)} desynced frames.")

    prev_flow_1, prev_flow_2 = None, None
    all_rows = []

    pd.DataFrame(columns=CSV_COLUMNS).to_csv(output_csv_path, index=False)
    print(f"[INIT] Output CSV initialized at: {output_csv_path}")
    print(f"[EXECUTION] Processing frames for {video_name}...")

    for i in tqdm(range(len(valid_keys)), desc=f"Extracting {video_name}"):
        actual_frame_num = valid_keys[i]
        frame_path = frame_paths[i]
        flow_path = flow_paths[i]
        
        try:
            frame = cv2.imread(frame_path)
            if frame is None:
                raise ValueError("OpenCV returned None.")
            f_t0 = np.load(flow_path).astype(np.float32)
        except Exception as e:
            tqdm.write(f"\n[CORRUPTION WARNING] Dropping frame {actual_frame_num}. File error: {e}")
            continue
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  
        
        div, curl = calculate_spatial_kinematics(f_t0)
        jerk = calculate_jerk(f_t0, prev_flow_1, prev_flow_2) if (prev_flow_1 is not None and prev_flow_2 is not None) else np.zeros_like(div)

        current_phase = "Idle"
        if labels_df is not None:
            phase = labels_df[labels_df['Frame'] == actual_frame_num]['Phase'].values
            if len(phase) > 0: current_phase = phase[0]

        # YOLO Tracking
        results = yolo_model.track(frame, persist=True, verbose=False)[0]
        
        if results.boxes is not None and len(results.boxes) > 0:
            boxes = results.boxes.xyxy.cpu().numpy()
            classes = results.boxes.cls.cpu().numpy()
            
            # --- ADDED: Safe Track ID Extraction ---
            # If a track is new/unconfirmed, YOLO might return None for the ID. We fallback to -1.
            if results.boxes.id is not None:
                track_ids = results.boxes.id.int().cpu().numpy()
            else:
                track_ids = [-1] * len(boxes)
            
            for box_idx, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box)
                tool_name = yolo_model.names[int(classes[box_idx])]
                tool_id = int(track_ids[box_idx]) # Get the specific ID for this box
                
                if tool_name not in TOOL_CLASSES:
                    continue 

                sam_results = sam_predictor(frame_rgb, bboxes=[x1, y1, x2, y2], verbose=False)[0]
                
                if sam_results.masks is not None and len(sam_results.masks.data) > 0:
                    raw_mask = sam_results.masks.data[0].cpu().numpy()
                    if raw_mask.shape != (frame.shape[0], frame.shape[1]):
                        tool_mask = cv2.resize(raw_mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
                    else:
                        tool_mask = raw_mask.astype(np.uint8)
                else:
                    tool_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                
                if actual_frame_num % 5000 == 0:
                    debug_frame = frame.copy()
                    debug_frame[tool_mask == 1] = [0, 0, 255] 
                    cv2.imwrite(f"debug_frame_{actual_frame_num}_tool_{tool_name}_id_{tool_id}.jpg", debug_frame)
                
                # Environment Boolean Split
                dilation_margin = 50
                env_x1, env_y1 = max(0, x1 - dilation_margin), max(0, y1 - dilation_margin)
                env_x2, env_y2 = min(frame.shape[1], x2 + dilation_margin), min(frame.shape[0], y2 + dilation_margin)
                
                env_mask = np.zeros_like(tool_mask)
                env_mask[env_y1:env_y2, env_x1:env_x2] = 1
                env_mask[tool_mask == 1] = 0 
                
                # Extract Physics Arrays
                t_jerk = extract_physics(tool_mask, jerk)
                t_div = extract_physics(tool_mask, div)
                t_curl = extract_physics(tool_mask, curl)
                
                e_jerk = extract_physics(env_mask, jerk)
                e_div = extract_physics(env_mask, div)
                e_curl = extract_physics(env_mask, curl)
                
                # --- ADDED: tool_id injected into row_data ---
                row_data = {
                    "video_name": video_name,
                    "frame_idx": actual_frame_num,
                    "tool_id": tool_id,
                    "tool_label": tool_name,
                    "phase_label": current_phase
                }
                
                metrics_data = t_jerk + t_div + t_curl + e_jerk + e_div + e_curl
                for col_name, val in zip(PHYSICS_METRICS, metrics_data):
                    row_data[col_name] = val
                    
                all_rows.append(row_data)

        prev_flow_2 = prev_flow_1
        prev_flow_1 = f_t0
        
        if (i + 1) % 500 == 0 or i == len(valid_keys) - 1:
            if len(all_rows) > 0:
                chunk_df = pd.DataFrame(all_rows)
                chunk_df = chunk_df[CSV_COLUMNS] 
                chunk_df.to_csv(output_csv_path, mode='a', header=False, index=False)
                all_rows = [] 
            
    print(f"\n[SUCCESS] MIL Matrix extraction complete. Saved to: {output_csv_path}")

# --- EXECUTION ---
#extract_video_physics(
#     video_name="videoXX",
#     frames_dir="frames_direct",
#     flow_dir="flow_direct",
#     phase_labels_csv="videoXX-phase.txt", 
#     yolo_model_path="weights.pt",
#     output_csv_path="output.csv"
#)