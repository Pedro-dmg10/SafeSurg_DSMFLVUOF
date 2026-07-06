# Decomposing Spatiotemporal Motion Features in Laparoscopic Video Using Optical Flow

This repository contains the official implementation of our texture-invariant surgical workflow analysis framework. The system systematically decomposes raw, sensorless laparoscopic optical flow into structured kinematic profiles to model tool-tissue interactions.

## Pipeline Architecture

The framework decouples surgical workflow analysis into two stages:

1. **Motion Feature Extraction:** Computes dense optical flow fields and isolates instrument/environment zones to extract 54-dimensional physical vectors (Divergence, Curl, Jerk metrics).
2. **Spatial-Temporal Structuring:** Processes variable tool bags using an Attention-based Multiple Instance Learning (MIL) module, followed by an LSTM sequence network for surgical phase recognition.

---

### 3. Model Weights Setup

Before running the pipeline, you must ensure the tracking and segmentation weights are placed in your working directory:

1. **YOLOv8 Weights:** Place your fine-tuned tool detection checkpoint (referenced as `weights.pt` in the script configuration) into your local folder.
2. **SAM 2.1 Weights:** The base model checkpoint (`sam2.1_b.pt`) will automatically download via the `ultralytics` framework upon your first execution. If running offline, manually download `sam2.1_b.pt` from the Ultralytics asset repository and place it in the project root directory.

## Installation

To replicate this environment and run the extraction pipeline, execute the following command from the root directory to fetch the missing FlowFormer++ dependencies:

```bash
git clone https://github.com/XiaoyuShi97/FlowFormerPlusPlus.git src/extraction/FlowFormer
```

## Prerequisites

Ensure you have a CUDA-capable GPU environment configured. Install the Python dependencies:

```bash
pip install -r requirements.txt
```
