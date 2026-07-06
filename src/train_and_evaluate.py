import os
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report

# --- SPEED OPTIMIZATION ---
torch.backends.cudnn.benchmark = True

class SurgicalSequenceDataset(Dataset):
    def __init__(self, file_paths, phase_encoder=None, tool_encoder=None, scaler=None, seq_len=25):
        self.sequences = []
        self.seq_len = seq_len
        self.phase_encoder = phase_encoder
        self.tool_encoder = tool_encoder
        self.scaler = scaler
        
        print(f"    -> [DATASET] Ingesting {len(file_paths)} CSV files into RAM...")
        dfs = []
        for f in file_paths:
            df = pd.read_csv(f)
            if not df.empty: 
                dfs.append(df)
            
        if not dfs: return
        combined_df = pd.concat(dfs, ignore_index=True)
        print(f"    -> [DATASET] Total rows loaded: {len(combined_df)}")
        
        print("    -> [DATASET] Applying Label Encoders...")
        if self.phase_encoder is None:
            self.phase_encoder = LabelEncoder()
            combined_df['phase_encoded'] = self.phase_encoder.fit_transform(combined_df['phase_label'])
        else:
            combined_df['phase_encoded'] = self.phase_encoder.transform(combined_df['phase_label'])
            
        if self.tool_encoder is None:
            self.tool_encoder = LabelEncoder()
            combined_df['tool_encoded'] = self.tool_encoder.fit_transform(combined_df['tool_label'])
        else:
            combined_df['tool_encoded'] = self.tool_encoder.transform(combined_df['tool_label'])
            
        print("    -> [DATASET] Normalizing Motion Features...")
        
        exclude_cols = [
            'video_name', 'frame_idx', 'tool_id', 'tool_label', 'phase_label', 
            'phase_encoded', 'tool_encoded', 'x1', 'y1', 'x2', 'y2'
        ]
        feature_cols = [c for c in combined_df.columns if c not in exclude_cols]
        
        if len(feature_cols) != 54:
            raise ValueError(f"FATAL: The network requires exactly 54 physics features, but extracted {len(feature_cols)}. Check your CSV structure.")
        
        if self.scaler is None:
            self.scaler = StandardScaler()
            combined_df[feature_cols] = self.scaler.fit_transform(combined_df[feature_cols])
        else:
            combined_df[feature_cols] = self.scaler.transform(combined_df[feature_cols])

        if 'tool_id' in combined_df.columns:
            combined_df['tool_id'] = combined_df['tool_id'].fillna(-1).astype(int)
        else:
            combined_df['tool_id'] = -1

        print(f"    -> [DATASET] Sorting chronologically and building {seq_len}-frame sequences...")
        combined_df.sort_values(['video_name', 'frame_idx'], inplace=True)
        
        all_phase_labels = []
        all_tool_labels = []

        video_groups = combined_df.groupby('video_name')
        
        for vid_name, v_group in video_groups:
            frame_groups = v_group.groupby('frame_idx')
            vid_bags = []
            
            for _, f_group in frame_groups:
                features = f_group[feature_cols].values.astype(np.float32)
                phase_label = f_group['phase_encoded'].iloc[0]
                tool_labels = f_group['tool_encoded'].values.astype(np.int64)
                tool_ids = f_group['tool_id'].values.astype(np.int64)
                
                all_tool_labels.extend(tool_labels)
                vid_bags.append({
                    'features': torch.tensor(features),       
                    'phase_label': torch.tensor(phase_label, dtype=torch.long), 
                    'tool_labels': torch.tensor(tool_labels, dtype=torch.long),
                    'tool_ids': torch.tensor(tool_ids, dtype=torch.long),
                    'frame_idxs': torch.tensor(f_group['frame_idx'].values, dtype=torch.long)
                })
                
            for i in range(len(vid_bags) - self.seq_len + 1):
                seq = vid_bags[i : i + self.seq_len]
                self.sequences.append(seq)
                all_phase_labels.append(seq[-1]['phase_label'].item())

        self.num_phases = len(self.phase_encoder.classes_)
        self.num_tools = len(self.tool_encoder.classes_)
        
        self.raw_phase_array = np.array(all_phase_labels)
        self.raw_tool_array = np.array(all_tool_labels)
        
        print(f"    -> [DATASET] SUCCESS: {len(self.sequences)} Temporal Sequences packaged.")

    def __len__(self): return len(self.sequences)
    def __getitem__(self, idx): return self.sequences[idx]


def seq_pad_collate(batch):
    B = len(batch)
    S = len(batch[0])
    
    flat_bags = [bag for seq in batch for bag in seq]
    
    features = [item['features'] for item in flat_bags]
    tool_labels = [item['tool_labels'] for item in flat_bags]
    tool_ids = [item['tool_ids'] for item in flat_bags]
    
    phase_labels = torch.stack([seq[-1]['phase_label'] for seq in batch])
    
    padded_features = nn.utils.rnn.pad_sequence(features, batch_first=True, padding_value=0.0)
    padded_tool_labels = nn.utils.rnn.pad_sequence(tool_labels, batch_first=True, padding_value=-100)
    padded_tool_ids = nn.utils.rnn.pad_sequence(tool_ids, batch_first=True, padding_value=-1)
    
    frame_idxs = [item['frame_idxs'] for item in flat_bags]
    padded_frame_idxs = nn.utils.rnn.pad_sequence(frame_idxs, batch_first=True, padding_value=-1)
    
    lengths = torch.tensor([f.size(0) for f in features])
    mask = torch.arange(padded_features.size(1))[None, :] < lengths[:, None]
    
    return padded_features, phase_labels, padded_tool_labels, padded_tool_ids, padded_frame_idxs, mask, B, S


class MultiTaskAttentionLSTM(nn.Module):
    def __init__(self, input_dim=54, hidden_dim=128, lstm_hidden=64, num_phases=7, num_tools=7):
        super(MultiTaskAttentionLSTM, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )
        self.tool_classifier = nn.Sequential(nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, num_tools))
        self.attention = nn.Sequential(nn.Linear(hidden_dim, 64), nn.Tanh(), nn.Linear(64, 1))
        
        self.lstm = nn.LSTM(input_size=hidden_dim, hidden_size=lstm_hidden, num_layers=1, batch_first=True)
        self.phase_classifier = nn.Sequential(nn.Linear(lstm_hidden, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, num_phases))

    def forward(self, bag_features, mask, batch_size, seq_len):
        H = self.encoder(bag_features) 
        tool_logits = self.tool_classifier(H) 
        
        A_raw = self.attention(H).squeeze(-1) 
        A_raw = A_raw.masked_fill(~mask, -1e4) 
        A_weights = F.softmax(A_raw, dim=1).unsqueeze(1) 
        
        M = torch.bmm(A_weights, H).squeeze(1) 
        
        M_seq = M.view(batch_size, seq_len, -1) 
        lstm_out, _ = self.lstm(M_seq) 
        last_out = lstm_out[:, -1, :] 
        phase_logits = self.phase_classifier(last_out) 
        
        return phase_logits, tool_logits, A_weights


def run_lovo_temporal_validation(csv_directory, epochs=15, batch_size=128, seq_len=25):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[SYSTEM] Deploying onto: {device}")
    
    all_csvs = sorted(glob.glob(os.path.join(csv_directory, "*.csv")))
    if len(all_csvs) < 2:
        raise ValueError(f"FATAL: Found {len(all_csvs)} CSVs. Need at least 2 for LOVO.")
    
    global_phase_results = []
    global_tool_results = []

    for test_idx, test_file in enumerate(all_csvs):
        test_video_name = os.path.basename(test_file)
        train_files = [f for i, f in enumerate(all_csvs) if i != test_idx]
        
        print(f"\nFOLD {test_idx + 1}/{len(all_csvs)} | TEST TARGET: {test_video_name}")
        
        train_dataset = SurgicalSequenceDataset(train_files, seq_len=seq_len)
        test_dataset = SurgicalSequenceDataset(
            [test_file], 
            phase_encoder=train_dataset.phase_encoder, 
            tool_encoder=train_dataset.tool_encoder,
            scaler=train_dataset.scaler,
            seq_len=seq_len
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=seq_pad_collate, num_workers=4, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=seq_pad_collate, num_workers=4, pin_memory=True)

        model = MultiTaskAttentionLSTM(num_phases=train_dataset.num_phases, num_tools=train_dataset.num_tools).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
        
        scaler = torch.cuda.amp.GradScaler()
        
        p_weights = compute_class_weight('balanced', classes=np.unique(train_dataset.raw_phase_array), y=train_dataset.raw_phase_array)
        t_weights = compute_class_weight('balanced', classes=np.unique(train_dataset.raw_tool_array), y=train_dataset.raw_tool_array)
        
        phase_criterion = nn.CrossEntropyLoss(weight=torch.tensor(p_weights, dtype=torch.float).to(device))
        tool_criterion = nn.CrossEntropyLoss(weight=torch.tensor(t_weights, dtype=torch.float).to(device), ignore_index=-100)

        for epoch in range(epochs):
            model.train()
            pbar = tqdm(train_loader, desc=f"FOLD {test_idx+1} | Epoch {epoch+1:02d}/{epochs}", leave=False)
            
            for bag, phase_label, tool_labels, tool_ids, frame_idxs, mask, B, S in pbar:
                bag, phase_label, tool_labels, mask = bag.to(device), phase_label.to(device), tool_labels.to(device), mask.to(device)
                
                optimizer.zero_grad(set_to_none=True)
                
                with torch.cuda.amp.autocast():
                    phase_logits, tool_logits, _ = model(bag, mask, B, S)
                    loss_phase = phase_criterion(phase_logits, phase_label)
                    loss_tool = tool_criterion(tool_logits.view(-1, train_dataset.num_tools), tool_labels.view(-1))
                    loss_total = loss_phase + loss_tool
                
                scaler.scale(loss_total).backward()
                scaler.step(optimizer)
                scaler.update()
                
                pbar.set_postfix({'P_Loss': f"{loss_phase.item():.2f}", 'T_Loss': f"{loss_tool.item():.2f}"})
                
            scheduler.step()
            
        weight_filename = f"lstm_fold_{test_idx + 1}_{test_video_name.replace('.csv', '')}.pth"
        torch.save(model.state_dict(), weight_filename)
        print(f"\n[SAVED] Fold {test_idx + 1} weights successfully stored at: {weight_filename}")
                
        model.eval()
        with torch.no_grad():
            for bag, phase_label, tool_labels, tool_ids, frame_idxs, mask, B, S in test_loader:
                bag, phase_label, tool_labels, mask = bag.to(device), phase_label.to(device), tool_labels.to(device), mask.to(device)
                
                with torch.cuda.amp.autocast():
                    phase_logits, tool_logits, _ = model(bag, mask, B, S)
                
                p_preds = torch.argmax(phase_logits, 1).cpu().numpy()
                p_trues = phase_label.cpu().numpy()
                for t, p in zip(p_trues, p_preds):
                    global_phase_results.append({'fold': test_idx, 'true': t, 'pred': p})

                # Stretch the phase labels [B] to match the flattened frames [B*S]
                expanded_p_trues = phase_label.repeat_interleave(S)
                # Stretch them again to match the max tools per frame [B*S, max_tools]
                expanded_p_trues = expanded_p_trues.unsqueeze(1).expand(-1, mask.size(1))
                
                # Now the phase tensor perfectly mirrors the tool mask shape
                valid_p_trues = expanded_p_trues[mask].cpu().numpy()
                decoded_phases = train_dataset.phase_encoder.inverse_transform(valid_p_trues)

                t_preds = torch.argmax(tool_logits, 2)[mask].cpu().numpy()
                t_trues = tool_labels[mask].cpu().numpy()
                t_ids = tool_ids[mask.cpu()].numpy()
                f_idxs = frame_idxs[mask.cpu()].numpy()
                
                # Map keys correctly for the analysis script
                for true_val, pred_val, track_id, f_idx, true_phase in zip(t_trues, t_preds, t_ids, f_idxs, decoded_phases):
                    global_tool_results.append({
                        'fold': test_idx, 
                        'frame_idx': f_idx,
                        'tool_id': track_id, 
                        'true_tool': true_val, 
                        'pred_tool': pred_val,
                        'true_phase': true_phase
                    })

    # --- TEMPORAL SMOOTHING (THE HYBRID TIERED APPROACH) ---
    print("\n    -> [SYSTEM] Applying Hybrid Temporal Smoothing (Phases: 15s | Ghost Noise: 3s | Tracks: Locked)...")
    
    df_phase = pd.DataFrame(global_phase_results)
    df_tools = pd.DataFrame(global_tool_results)
    
    # 1. PHASE SMOOTHING: 125 Frames (5 Seconds)
    df_phase['smoothed_pred'] = df_phase.groupby('fold')['pred'].transform(
        lambda x: x.rolling(window=125, center=True, min_periods=1).median().astype(int)
    )
    
    # 2. TOOL SMOOTHING (DUAL-TIER LOGIC)
    df_tools['smoothed_pred'] = df_tools['pred_tool']
    
    valid_track_mask = df_tools['tool_id'] != -1
    ghost_mask = df_tools['tool_id'] == -1
    
    if ghost_mask.any():
        df_tools.loc[ghost_mask, 'smoothed_pred'] = df_tools[ghost_mask].groupby('fold')['pred_tool'].transform(
            lambda x: x.rolling(window=75, center=True, min_periods=1).median().astype(int)
        )
        
    if valid_track_mask.any():
        df_tools.loc[valid_track_mask, 'smoothed_pred'] = df_tools[valid_track_mask].groupby(['fold', 'tool_id'])['pred_tool'].transform(
            lambda x: x.mode()[0] 
        )
    
    print("\n FINAL SYSTEM PERFORMANCE ACROSS ALL 12 FOLDS ")
    
    print("\n[1] SURGICAL PHASE PREDICTION (RAW / NO SMOOTHING)")
    print(classification_report(df_phase['true'], df_phase['pred'], target_names=train_dataset.phase_encoder.classes_))

    print("\n[2] SURGICAL PHASE PREDICTION (WITH 15-SECOND POST-SMOOTHING)")
    print(classification_report(df_phase['true'], df_phase['smoothed_pred'], target_names=train_dataset.phase_encoder.classes_))
    
    print("\n[3] BLIND TOOL PREDICTION (RAW / NO SMOOTHING)")
    print(classification_report(df_tools['true_tool'], df_tools['pred_tool'], target_names=train_dataset.tool_encoder.classes_))
    
    print("\n[4] BLIND TOOL PREDICTION (HYBRID TIERED SMOOTHING)")
    print(classification_report(df_tools['true_tool'], df_tools['smoothed_pred'], target_names=train_dataset.tool_encoder.classes_))

    # --- THE CRITICAL DATA EXPORT ---
    csv_out_path = "lstm_evaluation_results.csv"
    df_tools.to_csv(csv_out_path, index=False)
    print(f"\n[SYSTEM] Successfully exported evaluation tracking data to {csv_out_path}")

if __name__ == "__main__":
    # Replace with the local path to the directory containing your 12 motion CSV files
    csv_directory = "./data/motion_features/" 
    
    if os.path.exists(csv_directory):
        run_lovo_temporal_validation(csv_directory, epochs=15, batch_size=128, seq_len=25)
    else:
        print(f"[ERROR] Target directory not found: {csv_directory}. Please update the path in train_and_evaluate.py.")