"""
Attentive-SubTab: Self-Supervised Subset Encoding for Transactional Fraud Detection.

Reference implementation used to produce the results reported in the paper.
Partitions the input feature vector into K=4 overlapping subsets, pretrains a
dedicated encoder-decoder pair per subset with swap-noise reconstruction (SSL),
then fine-tunes a multi-head self-attention fusion module + classifier head
with a class-ratio weighted BCE loss and F1-optimal threshold selection.

Expects a preprocessed, chronologically split dataset cached as an .npz file
at DATA_CACHE_PATH (X_train, X_test, y_train, y_test). Runs the full pipeline
across the 9 seeds used in the paper (42-50) and writes a mean +/- std summary
plus the best-seed predictions to ./results/.

Usage:
    python attentive_subtab.py
"""

import os
import time
import copy
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, average_precision_score, precision_recall_curve

# ==========================================
# 1) CONFIGURATION & ARGUMENTS
# ==========================================
parser = argparse.ArgumentParser(description="Attentive-SubTab Fraud Detection")
parser.add_argument("--no-attention", action="store_true", help="Use mean pooling instead of MHSA")
parser.add_argument("--no-pretrain", action="store_true", help="Skip the SSL pretraining phase")
parser.add_argument("--no-swap-noise", action="store_true", help="Disable swap noise during pretraining")
args = parser.parse_args()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_CACHE_PATH = os.path.join(BASE_DIR, "attentive_subtab_dataset.npz")
RESULTS_PATH = os.path.join(BASE_DIR, "model_evaluation_results.npz")
SUMMARY_PATH = os.path.join(BASE_DIR, "9_seed_summary.txt")

SEEDS_9 = [42, 43, 44, 45, 46, 47, 48, 49, 50]
BATCH_SIZE = 1024
EPOCHS_PRETRAIN = 0 if args.no_pretrain else 5
EPOCHS_FINETUNE = 50
LEARNING_RATE = 1e-3
HIDDEN_DIM = 1024
N_SUBSETS = 4
MASKING_RATIO = 0.0 if args.no_swap_noise else 0.2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ==========================================
# 2) DATA AUGMENTATION (SSL NOISE)
# ==========================================
def apply_swap_noise(x: torch.Tensor, masking_ratio: float) -> torch.Tensor:
    if masking_ratio <= 0: return x
    B, F = x.shape
    mask = torch.rand((B, F), device=x.device) < masking_ratio
    if not mask.any(): return x
    rand_rows = torch.randint(0, B, (B, F), device=x.device)
    col_idx = torch.arange(F, device=x.device).unsqueeze(0).expand(B, F)
    x_swapped = x.clone()
    x_swapped[mask] = x[rand_rows[mask], col_idx[mask]]
    return x_swapped


# ==========================================
# 3) ARCHITECTURE COMPONENTS
# ==========================================
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU()
        )

    def forward(self, x): return self.net(x)


class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.BatchNorm1d(latent_dim * 2),
            nn.LeakyReLU(),
            nn.Linear(latent_dim * 2, output_dim)
        )

    def forward(self, x): return self.net(x)


class AttentionFusion(nn.Module):
    def __init__(self, latent_dim, num_heads=4, use_attention=True):
        super().__init__()
        self.use_attention = use_attention
        if self.use_attention:
            self.attention = nn.MultiheadAttention(embed_dim=latent_dim, num_heads=num_heads, batch_first=True)
            self.layer_norm = nn.LayerNorm(latent_dim)

    def forward(self, x_stack):
        if not self.use_attention:
            return x_stack.mean(dim=1)

        attn_out, _ = self.attention(x_stack, x_stack, x_stack)
        out = self.layer_norm(x_stack + attn_out)
        return out.mean(dim=1)


class ClassifierHead(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1)
        )

    def forward(self, x): return self.net(x).squeeze(-1)


# ==========================================
# 4) SINGLE SEED PIPELINE
# ==========================================
def run_one_seed(seed, X_train_full, X_test, y_train_full, y_test, subset_indices_t):
    torch.manual_seed(seed)
    np.random.seed(seed)

    # --- TIME-AWARE VALIDATION SPLIT ---
    split_idx = int(len(X_train_full) * 0.8)
    X_train_sub, y_train_sub = X_train_full[:split_idx], y_train_full[:split_idx]
    X_val, y_val = X_train_full[split_idx:], y_train_full[split_idx:]

    num_neg = (y_train_sub == 0).sum()
    num_pos = (y_train_sub == 1).sum()
    imbalance_ratio = num_neg / max(1, num_pos)
    print(f"  -> Detected Imbalance Ratio: {imbalance_ratio:.2f} (Weighting minority class...)")

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train_sub).float(), torch.from_numpy(y_train_sub).float()),
        batch_size=BATCH_SIZE, shuffle=True
    )

    subset_size = subset_indices_t[0].shape[0]

    # --- INTEGRATED FROM CODE 1: INDEPENDENT ENCODERS/DECODERS ---
    encoders = nn.ModuleList([Encoder(subset_size, HIDDEN_DIM) for _ in range(N_SUBSETS)]).to(device)
    decoders = nn.ModuleList([Decoder(HIDDEN_DIM // 2, subset_size) for _ in range(N_SUBSETS)]).to(device)

    fusion_module = AttentionFusion(HIDDEN_DIM // 2, use_attention=not args.no_attention).to(device)
    classifier = ClassifierHead(HIDDEN_DIM // 2).to(device)

    opt_pretrain = optim.AdamW(list(encoders.parameters()) + list(decoders.parameters()), lr=LEARNING_RATE,
                               weight_decay=1e-4)
    opt_finetune = optim.AdamW(
        list(encoders.parameters()) + list(fusion_module.parameters()) + list(classifier.parameters()),
        lr=LEARNING_RATE,
        weight_decay=1e-4
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt_finetune, mode='max', factor=0.5, patience=3, min_lr=1e-6)

    mse_loss = nn.MSELoss()
    bce_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([imbalance_ratio]).to(device))

    # PHASE 1: SSL PRE-TRAINING
    for epoch in range(EPOCHS_PRETRAIN):
        encoders.train()
        decoders.train()
        total_loss = 0
        for batch_x, _ in train_loader:
            batch_x = batch_x.to(device)
            subsets = [batch_x.index_select(1, idx) for idx in subset_indices_t]

            loss = 0
            # Route each subset to its dedicated encoder/decoder
            for i, s in enumerate(subsets):
                s_noisy = apply_swap_noise(s, MASKING_RATIO)
                z = encoders[i](s_noisy)
                recon = decoders[i](z)
                loss += mse_loss(recon, s)

            opt_pretrain.zero_grad()
            loss.backward()
            opt_pretrain.step()
            total_loss += loss.item()

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  [Pre-train] Epoch {epoch + 1}/{EPOCHS_PRETRAIN} | Loss: {total_loss / len(train_loader):.4f}")

    # PHASE 2: FINE-TUNING
    best_val_auc = 0.0
    best_thresh = 0.5
    best_model_state = None

    for epoch in range(EPOCHS_FINETUNE):
        encoders.train()
        fusion_module.train()
        classifier.train()
        total_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            subsets = [batch_x.index_select(1, idx) for idx in subset_indices_t]
            # Route subsets through dedicated encoders
            latents = [encoders[i](s) for i, s in enumerate(subsets)]
            stacked_latents = torch.stack(latents, dim=1)

            z = fusion_module(stacked_latents)
            logits = classifier(z)

            loss = bce_loss(logits, batch_y)
            opt_finetune.zero_grad()
            loss.backward()
            opt_finetune.step()
            total_loss += loss.item()

        # --- VALIDATION STEP ---
        encoders.eval()
        fusion_module.eval()
        classifier.eval()
        with torch.no_grad():
            X_val_t = torch.from_numpy(X_val).float().to(device)
            subsets_val = [X_val_t.index_select(1, idx) for idx in subset_indices_t]
            latents_val = [encoders[i](s) for i, s in enumerate(subsets_val)]
            stacked_latents_val = torch.stack(latents_val, dim=1)
            z_val = fusion_module(stacked_latents_val)
            logits_val = classifier(z_val)
            probs_val = torch.sigmoid(logits_val).cpu().numpy()

            val_auc = roc_auc_score(y_val, probs_val)
            val_pr_auc = average_precision_score(y_val, probs_val)

        scheduler.step(val_pr_auc)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_state = {
                'encoders': copy.deepcopy(encoders.state_dict()),
                'fusion': copy.deepcopy(fusion_module.state_dict()),
                'classifier': copy.deepcopy(classifier.state_dict())
            }

            precision_curve, recall_curve, thresholds = precision_recall_curve(y_val, probs_val)
            fscore = (2 * precision_curve * recall_curve) / (precision_curve + recall_curve + 1e-8)
            ix = np.argmax(fscore)
            best_thresh = thresholds[ix] if ix < len(thresholds) else 0.5

        if (epoch + 1) % 5 == 0 or epoch == 0:
            current_lr = opt_finetune.param_groups[0]['lr']
            print(
                f"  [Fine-tune] Epoch {epoch + 1}/{EPOCHS_FINETUNE} | Loss: {total_loss / len(train_loader):.4f} | Val AUROC: {val_auc:.4f} | LR: {current_lr:.1e}")

    print(f"  -> Best Validation Threshold Found: {best_thresh:.4f}")

    # Load the best model weights
    encoders.load_state_dict(best_model_state['encoders'])
    fusion_module.load_state_dict(best_model_state['fusion'])
    classifier.load_state_dict(best_model_state['classifier'])

    # PHASE 3: EVALUATION (TRAIN SET)
    encoders.eval()
    fusion_module.eval()
    classifier.eval()
    with torch.no_grad():
        X_train_sub_t = torch.from_numpy(X_train_sub).float().to(device)
        subsets_train = [X_train_sub_t.index_select(1, idx) for idx in subset_indices_t]
        latents_train = [encoders[i](s) for i, s in enumerate(subsets_train)]

        stacked_latents_train = torch.stack(latents_train, dim=1)
        z_train = fusion_module(stacked_latents_train)
        logits_train = classifier(z_train)

        probs_train = torch.sigmoid(logits_train).cpu().numpy()
        preds_train = (probs_train > best_thresh).astype(int)

    train_auroc = roc_auc_score(y_train_sub, probs_train)
    train_pr_auc = average_precision_score(y_train_sub, probs_train)
    train_precision = precision_score(y_train_sub, preds_train, zero_division=0)
    train_recall = recall_score(y_train_sub, preds_train)
    train_f1 = f1_score(y_train_sub, preds_train)
    train_macro_f1 = f1_score(y_train_sub, preds_train, average="macro")
    train_macro_prec = precision_score(y_train_sub, preds_train, average="macro", zero_division=0)

    train_metrics = [train_auroc, train_pr_auc, train_precision, train_recall, train_f1, train_macro_f1,
                     train_macro_prec]

    # PHASE 4: EVALUATION (TEST SET)
    with torch.no_grad():
        X_test_t = torch.from_numpy(X_test).float().to(device)
        subsets_test = [X_test_t.index_select(1, idx) for idx in subset_indices_t]
        latents_test = [encoders[i](s) for i, s in enumerate(subsets_test)]

        stacked_latents_test = torch.stack(latents_test, dim=1)
        z_test = fusion_module(stacked_latents_test)
        logits_test = classifier(z_test)

        probs = torch.sigmoid(logits_test).cpu().numpy()
        preds = (probs > best_thresh).astype(int)

    auroc = roc_auc_score(y_test, probs)
    pr_auc = average_precision_score(y_test, probs)
    precision = precision_score(y_test, preds, zero_division=0)
    recall = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    macro_f1 = f1_score(y_test, preds, average="macro")
    macro_prec = precision_score(y_test, preds, average="macro", zero_division=0)

    print(f"  -> Final Test AUROC for Seed {seed}: {auroc:.4f} | Recall: {recall:.4f}")

    test_metrics = [auroc, pr_auc, precision, recall, f1, macro_f1, macro_prec]
    return train_metrics, test_metrics, probs, preds


# ==========================================
# 5) MAIN DATA LOADING & 9-SEED LOOP
# ==========================================
def main():
    print("Loading 100% full pre-processed dataset from cache...")
    if not os.path.exists(DATA_CACHE_PATH):
        print(f"\nWARNING: Cache not found at {DATA_CACHE_PATH}.")
        return

    data = np.load(DATA_CACHE_PATH)

    X_train, X_test = data["X_train"], data["X_test"]
    y_train, y_test = data["y_train"], data["y_test"]

    n_features = X_train.shape[1]
    print(f"Training samples: {len(X_train)} | Testing samples: {len(X_test)}")

    subset_size = int(np.ceil(n_features / (N_SUBSETS / 2)))
    overlap = int(subset_size * 0.5)
    subset_indices = []
    start = 0
    for i in range(N_SUBSETS):
        idx = np.arange(start, start + subset_size)
        idx = idx[idx < n_features]
        if len(idx) < subset_size:
            idx = np.concatenate([idx, np.arange(0, subset_size - len(idx))])
        subset_indices.append(idx)
        start += (subset_size - overlap)

    subset_indices_t = [torch.from_numpy(idx).long().to(device) for idx in subset_indices]

    results_train_metrics = []
    results_test_metrics = []
    best_auroc = 0.0
    best_probs = None
    best_preds = None

    for seed in SEEDS_9:
        print(f"\n{'=' * 40}\nStarting Training for Seed: {seed}\n{'=' * 40}")
        train_metrics, test_metrics, probs, preds = run_one_seed(seed, X_train, X_test, y_train, y_test,
                                                                 subset_indices_t)
        results_train_metrics.append(train_metrics)
        results_test_metrics.append(test_metrics)

        if test_metrics[0] > best_auroc:
            best_auroc = test_metrics[0]
            best_probs = probs
            best_preds = preds

    # ==========================================
    # 6) MEAN & STD AGGREGATION
    # ==========================================
    res_train_arr = np.array(results_train_metrics)
    train_means = res_train_arr.mean(axis=0)
    train_stds = res_train_arr.std(axis=0)

    res_test_arr = np.array(results_test_metrics)
    test_means = res_test_arr.mean(axis=0)
    test_stds = res_test_arr.std(axis=0)

    summary_text = (
        "===== 9-SEED SUMMARY (Mean ± Std) =====\n\n"
        "--- TRAIN SET METRICS ---\n"
        f"AUROC:           {train_means[0]:.4f} ± {train_stds[0]:.4f}\n"
        f"PR-AUC:          {train_means[1]:.4f} ± {train_stds[1]:.4f}\n"
        f"Binary Precision:{train_means[2]:.4f} ± {train_stds[2]:.4f}\n"
        f"Binary Recall:   {train_means[3]:.4f} ± {train_stds[3]:.4f}\n"
        f"Binary F1:       {train_means[4]:.4f} ± {train_stds[4]:.4f}\n"
        f"Macro-F1:        {train_means[5]:.4f} ± {train_stds[5]:.4f}\n"
        f"Macro-Precision: {train_means[6]:.4f} ± {train_stds[6]:.4f}\n\n"
        "--- TEST SET METRICS ---\n"
        f"AUROC:           {test_means[0]:.4f} ± {test_stds[0]:.4f}\n"
        f"PR-AUC:          {test_means[1]:.4f} ± {test_stds[1]:.4f}\n"
        f"Binary Precision:{test_means[2]:.4f} ± {test_stds[2]:.4f}\n"
        f"Binary Recall:   {test_means[3]:.4f} ± {test_stds[3]:.4f}\n"
        f"Binary F1:       {test_means[4]:.4f} ± {test_stds[4]:.4f}\n"
        f"Macro-F1:        {test_means[5]:.4f} ± {test_stds[5]:.4f}\n"
        f"Macro-Precision: {test_means[6]:.4f} ± {test_stds[6]:.4f}\n"
    )

    print("\n" + summary_text)

    with open(SUMMARY_PATH, "w") as f:
        f.write(summary_text)
    print(f"Paper summary saved to: {SUMMARY_PATH}")

    np.savez_compressed(
        RESULTS_PATH,
        y_true=y_test,
        y_probs=best_probs,
        y_preds=best_preds,
        metrics=test_means[:5]
    )
    print(f"Visualizer data saved to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()