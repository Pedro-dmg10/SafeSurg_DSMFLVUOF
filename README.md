# Decomposing Spatiotemporal Motion Features in Laparoscopic Video Using Optical Flow

This repository contains the official implementation of our texture-invariant surgical workflow analysis framework. The system systematically decomposes raw, sensorless laparoscopic optical flow into structured kinematic profiles to model tool-tissue interactions.

## Pipeline Architecture
The framework decouples surgical workflow analysis into two stages:
1. **Motion Feature Extraction:** Computes dense optical flow fields and isolates instrument/environment zones to extract 54-dimensional physical vectors (Divergence, Curl, Jerk metrics).
2. **Spatial-Temporal Structuring:** Processes variable tool bags using an Attention-based Multiple Instance Learning (MIL) module, followed by an LSTM sequence network for surgical phase recognition.

---

## Installation

### Prerequisites
Ensure you have a CUDA-capable GPU environment configured. Clone the repository and install the dependencies:

```bash
pip install -r requirements.txt
