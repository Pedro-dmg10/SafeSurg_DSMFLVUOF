import sys
sys.path.append('core')

import argparse
import os
import numpy as np
import torch
from glob import glob
from torch.utils.data import Dataset, DataLoader

from configs.submissions import get_cfg
from core.FlowFormer import build_flowformer
from utils.utils import InputPadder
from utils import frame_utils

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class CholecDataset(Dataset):
    """Loads images on parallel CPU threads so the GPU never waits."""
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        f1, f2, npy_out_path = self.pairs[idx]
        
        # Read and convert directly to tensors
        image1 = frame_utils.read_gen(f1)
        image2 = frame_utils.read_gen(f2)

        image1 = np.array(image1)[..., :3].astype(np.uint8)
        image2 = np.array(image2)[..., :3].astype(np.uint8)

        # Output shape: [3, H, W]
        t1 = torch.from_numpy(image1).permute(2, 0, 1).float()
        t2 = torch.from_numpy(image2).permute(2, 0, 1).float()

        return t1, t2, npy_out_path


def build_model():
    print("Building FlowFormer++ model...")
    cfg = get_cfg()
    model = torch.nn.DataParallel(build_flowformer(cfg))
    cfg.model = "ckpt/sintel.pth"
    model.load_state_dict(torch.load(cfg.model))
    model.cuda()
    model.eval()
    return model


def run_batched_inference(model, dataloader):
    print(f"Starting batched inference on {len(dataloader.dataset)} pairs...")
    
    with torch.no_grad():
        for batch_idx, (img1_batch, img2_batch, npy_paths) in enumerate(dataloader):
            
            # Push the whole batch to GPU at once
            img1_batch = img1_batch.cuda(non_blocking=True)
            img2_batch = img2_batch.cuda(non_blocking=True)

            padder = InputPadder(img1_batch.shape)
            img1_batch, img2_batch = padder.pad(img1_batch, img2_batch)

            # GPU does the math
            flow_pre, _ = model(img1_batch, img2_batch)
            flow_pre = padder.unpad(flow_pre)

            # Bring the batch back to CPU
            flow_batch = flow_pre.permute(0, 2, 3, 1).cpu().numpy()

            # Save the vectors
            for i in range(len(npy_paths)):
                out_path = npy_paths[i]
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                np.save(out_path, flow_batch[i])
            
            if batch_idx % 10 == 0:
                print(f"Processed batch {batch_idx}/{len(dataloader)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", required=True)
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=8)
    args = parser.parse_args()

    video_name = os.path.basename(os.path.normpath(args.root_dir))
    npy_root = os.path.join(args.out_root, f"flow_vectors/{video_name}")

    print(f"Scanning {args.root_dir} for frames...")
    all_frames = sorted(
        glob(os.path.join(args.root_dir, "**", "*.jpg"), recursive=True) +
        glob(os.path.join(args.root_dir, "**", "*.png"), recursive=True)
    )
    
    if not all_frames:
        print(f"CRITICAL ERROR: No frames found in {args.root_dir}.")
        sys.exit(1)

    # Cross-boundary pairing
    img_pairs = []
    for i in range(len(all_frames) - 1):
        img_pairs.append((all_frames[i], all_frames[i+1]))

    valid_pairs = []
    for f1, f2 in img_pairs:
        rel_path = os.path.relpath(f1, args.root_dir)
        frame_name = os.path.splitext(os.path.basename(f1))[0]
        out_path = os.path.join(npy_root, os.path.dirname(rel_path), f"{frame_name}.npy")
        
        if not os.path.exists(out_path):
            valid_pairs.append((f1, f2, out_path))

    print(f"Found {len(img_pairs)} total pairs. {len(img_pairs) - len(valid_pairs)} already processed.")
    
    if len(valid_pairs) == 0:
        print("Everything is already processed. Exiting cleanly.")
        sys.exit(0)

    model = build_model()
    
    dataset = CholecDataset(valid_pairs)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)

    run_batched_inference(model, dataloader)