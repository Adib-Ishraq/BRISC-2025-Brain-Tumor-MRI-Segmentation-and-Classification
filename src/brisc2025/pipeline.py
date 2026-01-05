# -*- coding: utf-8 -*-
"""
BRISC2025 (CSE428) — Cleaned end-to-end pipeline

This file is a cleaned / de-duplicated version of your `final_28.py` notebook export:
- Stage-1: Binary tumor segmentation (U-Net or Attention U-Net)
- Stage-2: Tumor-type classification using the frozen Stage-1 encoder
- (Optional) Joint multi-task U-Net (seg + cls)
- (Optional) Compare 3 pretrained classifier backbones (MobileNetV2 / EfficientNetB0 / DenseNet121)

Designed to run in Kaggle with:
DATA_ROOT=/kaggle/input/brisc2025/brisc2025

You can toggle which parts to run in the CONFIG section at the bottom.
"""

# -------------------------
# Imports (one place only)
# -------------------------
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# pretrained backbones (optional)
import torchvision
from torchvision.models import (
    mobilenet_v2, MobileNet_V2_Weights,
    efficientnet_b0, EfficientNet_B0_Weights,
    densenet121, DenseNet121_Weights,
)

# -------------------------
# Reproducibility + device
# -------------------------
def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_everything(42)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================================================
# 1) Dataset helpers
# =========================================================
LABELS = ["glioma", "meningioma", "pituitary", "no_tumor"]
CODE_TO_LABEL = {"gl": "glioma", "me": "meningioma", "pi": "pituitary", "nt": "no_tumor"}
LABEL_TO_ID = {lab: i for i, lab in enumerate(LABELS)}
ID_TO_LABEL = {i: lab for lab, i in LABEL_TO_ID.items()}

def tumor_label_from_filename(fname: str) -> str:
    """
    Expected pattern (example):
      brisc2025_train_00001_gl_ax_t1.jpg
                         ^^^
    """
    parts = os.path.basename(fname).split("_")
    if len(parts) < 4:
        raise ValueError(f"Unexpected filename: {fname}")
    code = parts[3].lower()
    if code not in CODE_TO_LABEL:
        raise ValueError(f"Unknown tumor code '{code}' in filename: {fname}")
    return CODE_TO_LABEL[code]

def build_seg_df(img_dir: str, msk_dir: str) -> pd.DataFrame:
    """Absolute paths for segmentation images + masks + label."""
    recs = []
    for fname in sorted(os.listdir(img_dir)):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        base = os.path.splitext(fname)[0]
        img_path = os.path.join(img_dir, fname)
        msk_path = os.path.join(msk_dir, base + ".png")
        if not os.path.exists(msk_path):
            continue
        recs.append({
            "img_path": img_path,
            "msk_path": msk_path,
            "label": tumor_label_from_filename(fname),
        })
    return pd.DataFrame(recs)

def build_cls_df(img_dir: str, split_name: str, data_root: str) -> pd.DataFrame:
    """Relative image paths (to keep Kaggle paths clean) + label."""
    recs = []
    for fname in sorted(os.listdir(img_dir)):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        rel_img = os.path.join("segmentation_task", split_name, "images", fname)
        recs.append({"image_path": rel_img, "label": tumor_label_from_filename(fname)})
    # sanity: make sure paths exist
    for p in recs[:3]:
        full = os.path.join(data_root, p["image_path"])
        if not os.path.exists(full):
            raise FileNotFoundError(full)
    return pd.DataFrame(recs)

def stratified_split(df: pd.DataFrame, val_size: float = 0.2, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df, val_df = train_test_split(
        df, test_size=val_size, random_state=seed, stratify=df["label"]
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


# =========================================================
# 2) Datasets
# =========================================================
class BriscSegDataset(Dataset):
    """Returns (image[1,H,W], mask[1,H,W])"""
    def __init__(self, df: pd.DataFrame, img_size=(384, 384), augment: bool = False):
        self.df = df.reset_index(drop=True)
        self.img_size = img_size
        self.augment = augment

    def __len__(self): return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = Image.open(row["img_path"]).convert("L")
        msk = Image.open(row["msk_path"]).convert("L")

        img = TF.resize(img, self.img_size, interpolation=InterpolationMode.BILINEAR)
        msk = TF.resize(msk, self.img_size, interpolation=InterpolationMode.NEAREST)

        x = TF.to_tensor(img)  # [1,H,W]
        m = (np.array(msk, dtype=np.uint8) > 0).astype(np.float32)
        y = torch.from_numpy(m).unsqueeze(0)  # [1,H,W]

        if self.augment:
            if random.random() < 0.5:
                x = torch.flip(x, dims=[2]); y = torch.flip(y, dims=[2])  # hflip
            if random.random() < 0.5:
                x = torch.flip(x, dims=[1]); y = torch.flip(y, dims=[1])  # vflip

        return x, y

class BriscClsDatasetGray(Dataset):
    """Grayscale classification dataset -> (image[1,H,W], label_id)"""
    def __init__(self, df: pd.DataFrame, data_root: str, img_size=(384, 384), augment: bool = False):
        self.df = df.reset_index(drop=True)
        self.data_root = data_root
        self.img_size = img_size
        self.augment = augment
        self.jitter = T.ColorJitter(brightness=0.15, contrast=0.15)

    def __len__(self): return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.data_root, row["image_path"])
        label = LABEL_TO_ID[row["label"]]

        img = Image.open(img_path).convert("L")
        img = TF.resize(img, self.img_size, interpolation=InterpolationMode.BILINEAR)

        if self.augment:
            angle = random.uniform(-10, 10)
            img = TF.rotate(img, angle, interpolation=InterpolationMode.BILINEAR, fill=0)
            if random.random() < 0.5:
                img = TF.hflip(img)
            img = self.jitter(img)

        x = TF.to_tensor(img)  # [1,H,W]
        if self.augment and random.random() < 0.15:
            x = (x + 0.03 * torch.randn_like(x)).clamp(0, 1)

        return x, torch.tensor(label, dtype=torch.long)

class BriscClsDatasetRGB(Dataset):
    """RGB classification dataset for ImageNet-pretrained backbones."""
    def __init__(self, df: pd.DataFrame, data_root: str, transform: T.Compose):
        self.df = df.reset_index(drop=True)
        self.data_root = data_root
        self.transform = transform

    def __len__(self): return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.data_root, row["image_path"])
        y = LABEL_TO_ID[row["label"]]
        img = Image.open(img_path).convert("RGB")
        x = self.transform(img)
        return x, torch.tensor(y, dtype=torch.long)

class BriscMultiTaskDataset(Dataset):
    """Returns (image[1,H,W], mask[1,H,W], class_id) for joint training."""
    def __init__(self, df: pd.DataFrame, img_size=(384, 384), augment: bool = False):
        self.df = df.reset_index(drop=True)
        self.img_size = img_size
        self.augment = augment

    def __len__(self): return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = Image.open(row["img_path"]).convert("L")
        msk = Image.open(row["msk_path"]).convert("L")

        img = TF.resize(img, self.img_size, interpolation=InterpolationMode.BILINEAR)
        msk = TF.resize(msk, self.img_size, interpolation=InterpolationMode.NEAREST)

        x = TF.to_tensor(img)
        m = (np.array(msk, dtype=np.uint8) > 0).astype(np.float32)
        mask_t = torch.from_numpy(m).unsqueeze(0)
        cls_t = torch.tensor(LABEL_TO_ID[row["label"]], dtype=torch.long)

        if self.augment:
            if random.random() < 0.5:
                x = torch.flip(x, dims=[2]); mask_t = torch.flip(mask_t, dims=[2])
            if random.random() < 0.5:
                x = torch.flip(x, dims=[1]); mask_t = torch.flip(mask_t, dims=[1])

        return x, mask_t, cls_t


# =========================================================
# 3) Models
# =========================================================
class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x): return self.net(x)

class UNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, base=32):
        super().__init__()
        self.enc1 = DoubleConv(in_ch, base)
        self.enc2 = DoubleConv(base, base * 2)
        self.enc3 = DoubleConv(base * 2, base * 4)
        self.enc4 = DoubleConv(base * 4, base * 8)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(base * 8, base * 16)

        self.up4 = nn.ConvTranspose2d(base * 16, base * 8, 2, stride=2)
        self.dec4 = DoubleConv(base * 16, base * 8)

        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = DoubleConv(base * 8, base * 4)

        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = DoubleConv(base * 4, base * 2)

        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = DoubleConv(base * 2, base)

        self.out = nn.Conv2d(base, out_ch, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        d4 = self.up4(b); d4 = self.dec4(torch.cat([d4, e4], dim=1))
        d3 = self.up3(d4); d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3); d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2); d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.out(d1)  # logits [B,1,H,W]

class UNetEncoder(nn.Module):
    """Encoder + bottleneck of U-Net (used for Stage-2 classification)."""
    def __init__(self, in_ch=1, base=32):
        super().__init__()
        self.enc1 = DoubleConv(in_ch, base)
        self.enc2 = DoubleConv(base, base * 2)
        self.enc3 = DoubleConv(base * 2, base * 4)
        self.enc4 = DoubleConv(base * 4, base * 8)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(base * 8, base * 16)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        return b  # [B, base*16, H/16, W/16]

class EncoderClassifier(nn.Module):
    def __init__(self, encoder: nn.Module, feat_ch: int, num_classes: int = 4):
        super().__init__()
        self.encoder = encoder
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat_ch, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        feat = self.encoder(x)
        feat = self.gap(feat)
        return self.head(feat)

# ---- Attention U-Net (optional) ----
class AttentionGate(nn.Module):
    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()
        self.W_g = nn.Sequential(nn.Conv2d(F_g, F_int, 1, bias=True), nn.BatchNorm2d(F_int))
        self.W_x = nn.Sequential(nn.Conv2d(F_l, F_int, 1, bias=True), nn.BatchNorm2d(F_int))
        self.psi = nn.Sequential(nn.Conv2d(F_int, 1, 1, bias=True), nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, g):
        psi = self.relu(self.W_g(g) + self.W_x(x))
        alpha = self.psi(psi)
        return x * alpha

class AttentionUNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, base=32):
        super().__init__()
        self.enc1 = DoubleConv(in_ch, base)
        self.enc2 = DoubleConv(base, base * 2)
        self.enc3 = DoubleConv(base * 2, base * 4)
        self.enc4 = DoubleConv(base * 4, base * 8)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(base * 8, base * 16)

        self.up4 = nn.ConvTranspose2d(base * 16, base * 8, 2, stride=2)
        self.att4 = AttentionGate(base * 8, base * 8, base * 4)
        self.dec4 = DoubleConv(base * 16, base * 8)

        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.att3 = AttentionGate(base * 4, base * 4, base * 2)
        self.dec3 = DoubleConv(base * 8, base * 4)

        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.att2 = AttentionGate(base * 2, base * 2, base)
        self.dec2 = DoubleConv(base * 4, base * 2)

        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.att1 = AttentionGate(base, base, max(1, base // 2))
        self.dec1 = DoubleConv(base * 2, base)

        self.out = nn.Conv2d(base, out_ch, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        d4 = self.up4(b); e4a = self.att4(e4, d4); d4 = self.dec4(torch.cat([d4, e4a], dim=1))
        d3 = self.up3(d4); e3a = self.att3(e3, d3); d3 = self.dec3(torch.cat([d3, e3a], dim=1))
        d2 = self.up2(d3); e2a = self.att2(e2, d2); d2 = self.dec2(torch.cat([d2, e2a], dim=1))
        d1 = self.up1(d2); e1a = self.att1(e1, d1); d1 = self.dec1(torch.cat([d1, e1a], dim=1))
        return self.out(d1)

# ---- Multi-task U-Net (optional) ----
class MultiTaskUNet(nn.Module):
    """Single U-Net with seg head + cls head from bottleneck."""
    def __init__(self, in_ch=1, seg_out_ch=1, num_classes=4, base=32):
        super().__init__()
        self.enc1 = DoubleConv(in_ch, base)
        self.enc2 = DoubleConv(base, base * 2)
        self.enc3 = DoubleConv(base * 2, base * 4)
        self.enc4 = DoubleConv(base * 4, base * 8)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(base * 8, base * 16)

        # seg decoder
        self.up4 = nn.ConvTranspose2d(base * 16, base * 8, 2, stride=2)
        self.dec4 = DoubleConv(base * 16, base * 8)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = DoubleConv(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = DoubleConv(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = DoubleConv(base * 2, base)
        self.seg_out = nn.Conv2d(base, seg_out_ch, 1)

        # cls head
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.cls_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(base * 16, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        cls_logits = self.cls_head(self.gap(b))

        d4 = self.up4(b); d4 = self.dec4(torch.cat([d4, e4], dim=1))
        d3 = self.up3(d4); d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3); d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2); d1 = self.dec1(torch.cat([d1, e1], dim=1))
        seg_logits = self.seg_out(d1)
        return seg_logits, cls_logits


# =========================================================
# 4) Losses + Metrics (shared)
# =========================================================
_bce = nn.BCEWithLogitsLoss()
_ce  = nn.CrossEntropyLoss()

def dice_loss_soft(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    probs_f = probs.view(probs.size(0), -1)
    t_f = targets.view(targets.size(0), -1)
    inter = (probs_f * t_f).sum(dim=1)
    dice = (2 * inter + eps) / (probs_f.sum(dim=1) + t_f.sum(dim=1) + eps)
    return 1 - dice.mean()

def seg_loss_fn(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return _bce(logits, targets) + dice_loss_soft(logits, targets)

@torch.no_grad()
def seg_confusion_from_logits(logits: torch.Tensor, targets: torch.Tensor, thr: float = 0.5):
    probs = torch.sigmoid(logits)
    preds = (probs > thr).float()
    p = preds.view(-1)
    t = targets.view(-1)
    tp = (p * t).sum().item()
    fp = (p * (1 - t)).sum().item()
    fn = ((1 - p) * t).sum().item()
    tn = ((1 - p) * (1 - t)).sum().item()
    return tp, fp, fn, tn

def seg_metrics_from_confusion(tp, fp, fn, tn, eps: float = 1e-7):
    iou_fg = (tp + eps) / (tp + fp + fn + eps)
    iou_bg = (tn + eps) / (tn + fn + fp + eps)
    miou = 0.5 * (iou_fg + iou_bg)
    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    pixacc = (tp + tn + eps) / (tp + tn + fp + fn + eps)
    return float(miou), float(dice), float(pixacc)

def cls_metrics_from_confmat(cm: np.ndarray, eps: float = 1e-9):
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    prec = (tp + eps) / (tp + fp + eps)
    rec = (tp + eps) / (tp + fn + eps)
    f1 = (2 * prec * rec + eps) / (prec + rec + eps)
    acc = float(tp.sum() / (cm.sum() + eps))
    return acc, float(prec.mean()), float(rec.mean()), float(f1.mean())


# =========================================================
# 5) AMP helpers
# =========================================================
def make_amp():
    """Compatible autocast/scaler across PyTorch versions."""
    try:
        from torch.amp import autocast, GradScaler
        autocast_ctx = lambda: autocast(device_type="cuda", enabled=(DEVICE == "cuda"))
        scaler = GradScaler(enabled=(DEVICE == "cuda"))
        return autocast_ctx, scaler
    except Exception:
        from torch.cuda.amp import autocast as autocast_cuda
        from torch.cuda.amp import GradScaler
        autocast_ctx = lambda: autocast_cuda(enabled=(DEVICE == "cuda"))
        scaler = GradScaler(enabled=(DEVICE == "cuda"))
        return autocast_ctx, scaler

AUT, SCALER = make_amp()


# =========================================================
# 6) Train / Eval loops
# =========================================================
@torch.no_grad()
def eval_seg(model: nn.Module, loader: DataLoader):
    model.eval()
    total_loss, n = 0.0, 0
    tp = fp = fn = tn = 0.0

    for x, y in loader:
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)
        with AUT():
            logits = model(x)
            loss = seg_loss_fn(logits, y)
        total_loss += float(loss.item()); n += 1
        a, b, c, d = seg_confusion_from_logits(logits, y)
        tp += a; fp += b; fn += c; tn += d

    miou, dice, pix = seg_metrics_from_confusion(tp, fp, fn, tn)
    return total_loss / max(1, n), miou, dice, pix

def train_seg_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer):
    model.train()
    total_loss, n = 0.0, 0
    tp = fp = fn = tn = 0.0

    for x, y in loader:
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with AUT():
            logits = model(x)
            loss = seg_loss_fn(logits, y)

        SCALER.scale(loss).backward()
        SCALER.step(optimizer)
        SCALER.update()

        total_loss += float(loss.item()); n += 1
        a, b, c, d = seg_confusion_from_logits(logits.detach(), y)
        tp += a; fp += b; fn += c; tn += d

    miou, dice, pix = seg_metrics_from_confusion(tp, fp, fn, tn)
    return total_loss / max(1, n), miou, dice, pix

@torch.no_grad()
def eval_cls(model: nn.Module, loader: DataLoader, num_classes: int = 4):
    model.eval()
    total_loss, n = 0.0, 0
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)

    for x, y in loader:
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)
        logits = model(x)
        loss = _ce(logits, y)

        total_loss += float(loss.item()); n += 1
        preds = logits.argmax(dim=1).detach().cpu().numpy()
        trues = y.detach().cpu().numpy()
        for t, p in zip(trues, preds):
            cm[t, p] += 1

    acc, mp, mr, mf1 = cls_metrics_from_confmat(cm)
    return total_loss / max(1, n), acc, mp, mr, mf1

def train_cls_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, num_classes: int = 4):
    model.train()
    if hasattr(model, "encoder"):
        # If the encoder is frozen, keep BN stable
        model.encoder.eval()

    total_loss, n = 0.0, 0
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)

    for x, y in loader:
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = _ce(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item()); n += 1
        preds = logits.argmax(dim=1).detach().cpu().numpy()
        trues = y.detach().cpu().numpy()
        for t, p in zip(trues, preds):
            cm[t, p] += 1

    acc, mp, mr, mf1 = cls_metrics_from_confmat(cm)
    return total_loss / max(1, n), acc, mp, mr, mf1

@torch.no_grad()
def eval_joint(model: nn.Module, loader: DataLoader, lambda_cls: float = 0.5):
    model.eval()
    total_loss, n = 0.0, 0
    tp = fp = fn = tn = 0.0
    cm = np.zeros((len(LABELS), len(LABELS)), dtype=np.int64)

    for x, m, y in loader:
        x = x.to(DEVICE, non_blocking=True)
        m = m.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)

        with AUT():
            seg_logits, cls_logits = model(x)
            loss = seg_loss_fn(seg_logits, m) + lambda_cls * _ce(cls_logits, y)

        total_loss += float(loss.item()); n += 1

        a, b, c, d = seg_confusion_from_logits(seg_logits, m)
        tp += a; fp += b; fn += c; tn += d

        preds = cls_logits.argmax(dim=1).detach().cpu().numpy()
        trues = y.detach().cpu().numpy()
        for t, p in zip(trues, preds):
            cm[t, p] += 1

    seg_miou, seg_dice, seg_pix = seg_metrics_from_confusion(tp, fp, fn, tn)
    cls_acc, cls_mp, cls_mr, cls_f1 = cls_metrics_from_confmat(cm)
    return (
        total_loss / max(1, n),
        (seg_miou, seg_dice, seg_pix),
        (cls_acc, cls_mp, cls_mr, cls_f1),
    )

def train_joint_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, lambda_cls: float = 0.5):
    model.train()
    total_loss, n = 0.0, 0
    tp = fp = fn = tn = 0.0
    cm = np.zeros((len(LABELS), len(LABELS)), dtype=np.int64)

    for x, m, y in loader:
        x = x.to(DEVICE, non_blocking=True)
        m = m.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with AUT():
            seg_logits, cls_logits = model(x)
            loss = seg_loss_fn(seg_logits, m) + lambda_cls * _ce(cls_logits, y)

        SCALER.scale(loss).backward()
        SCALER.step(optimizer)
        SCALER.update()

        total_loss += float(loss.item()); n += 1

        a, b, c, d = seg_confusion_from_logits(seg_logits.detach(), m)
        tp += a; fp += b; fn += c; tn += d

        preds = cls_logits.argmax(dim=1).detach().cpu().numpy()
        trues = y.detach().cpu().numpy()
        for t, p in zip(trues, preds):
            cm[t, p] += 1

    seg_miou, seg_dice, seg_pix = seg_metrics_from_confusion(tp, fp, fn, tn)
    cls_acc, cls_mp, cls_mr, cls_f1 = cls_metrics_from_confmat(cm)
    return (
        total_loss / max(1, n),
        (seg_miou, seg_dice, seg_pix),
        (cls_acc, cls_mp, cls_mr, cls_f1),
    )


# =========================================================
# 7) Visualization helpers
# =========================================================
def _load_image_tensor_gray(image_path: str, img_size=(384, 384)):
    img_orig = Image.open(image_path).convert("L")
    img_proc = TF.resize(img_orig, img_size, interpolation=InterpolationMode.BILINEAR)
    x = TF.to_tensor(img_proc).unsqueeze(0)  # [1,1,H,W]
    return img_orig, img_proc, x

def _load_mask_np(mask_path: str, img_size=(384, 384)):
    m = Image.open(mask_path).convert("L")
    m = TF.resize(m, img_size, interpolation=InterpolationMode.NEAREST)
    return (np.array(m, dtype=np.uint8) > 0).astype(np.float32)

def _seg_metrics_np(pred_bin: np.ndarray, gt_bin: np.ndarray, eps: float = 1e-7):
    p = pred_bin.reshape(-1).astype(np.float32)
    t = gt_bin.reshape(-1).astype(np.float32)
    tp = (p * t).sum()
    fp = (p * (1 - t)).sum()
    fn = ((1 - p) * t).sum()
    tn = ((1 - p) * (1 - t)).sum()
    return seg_metrics_from_confusion(tp, fp, fn, tn, eps=eps)

@torch.no_grad()
def infer_separate(seg_model: nn.Module, cls_model: nn.Module, image_path: str, mask_path: Optional[str] = None,
                   img_size=(384, 384), thr: float = 0.5, save_path: Optional[str] = None):
    """Single-sample inference + 2x3 visualization for separate models."""
    seg_model.eval(); cls_model.eval()
    img_orig, img_proc, x = _load_image_tensor_gray(image_path, img_size)
    x = x.to(DEVICE)

    seg_logits = seg_model(x)
    seg_prob = torch.sigmoid(seg_logits)[0, 0].cpu().numpy()
    pred_mask = (seg_prob > thr).astype(np.float32)

    cls_logits = cls_model(x)
    pred_id = int(cls_logits.argmax(dim=1).item())
    pred_label = ID_TO_LABEL[pred_id]

    gt_mask = None
    miou = dice = pix = None
    if mask_path and os.path.exists(mask_path):
        gt_mask = _load_mask_np(mask_path, img_size)
        miou, dice, pix = _seg_metrics_np(pred_mask, gt_mask)

    # plot
    img_np = np.array(img_proc)
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    axes[0, 0].imshow(img_orig, cmap="gray"); axes[0, 0].set_title("Original image"); axes[0, 0].axis("off")
    axes[0, 1].imshow(gt_mask if gt_mask is not None else np.zeros_like(pred_mask), cmap="jet")
    axes[0, 1].set_title("GT mask" if gt_mask is not None else "GT mask (N/A)"); axes[0, 1].axis("off")

    axes[0, 2].imshow(img_np, cmap="gray")
    if gt_mask is not None:
        axes[0, 2].imshow(gt_mask, cmap="jet", alpha=0.4)
    axes[0, 2].set_title("Image + GT overlay"); axes[0, 2].axis("off")

    axes[1, 0].imshow(img_np, cmap="gray"); axes[1, 0].set_title("Processed image"); axes[1, 0].axis("off")
    axes[1, 1].imshow(pred_mask, cmap="jet"); axes[1, 1].set_title("Predicted mask"); axes[1, 1].axis("off")
    axes[1, 2].imshow(img_np, cmap="gray"); axes[1, 2].imshow(pred_mask, cmap="jet", alpha=0.4)
    axes[1, 2].set_title("Image + Pred overlay"); axes[1, 2].axis("off")

    miou_txt = "N/A" if miou is None else f"{miou:.3f}"
    dice_txt = "N/A" if dice is None else f"{dice:.3f}"
    fig.suptitle(f"Pred class: {pred_label} | mIoU={miou_txt} | Dice={dice_txt}", fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


# =========================================================
# 8) Train runners (Stage-1, Stage-2, Joint, Backbone compare)
# =========================================================
@dataclass
class Paths:
    data_root: str = "/kaggle/input/brisc2025/brisc2025"
    out_dir: str = "/kaggle/working"

    @property
    def seg_train_img_dir(self): return os.path.join(self.data_root, "segmentation_task/train/images")
    @property
    def seg_train_msk_dir(self): return os.path.join(self.data_root, "segmentation_task/train/masks")
    @property
    def seg_test_img_dir(self):  return os.path.join(self.data_root, "segmentation_task/test/images")
    @property
    def seg_test_msk_dir(self):  return os.path.join(self.data_root, "segmentation_task/test/masks")

def run_stage1_segmentation(paths: Paths,
                            model_type: str = "unet",
                            img_size=(384, 384),
                            batch_size: int = 8,
                            epochs: int = 40,
                            lr: float = 1e-3,
                            early_stop_patience: int = 7,
                            ckpt_name: str = "stage1_seg_best.pth"):
    os.makedirs(paths.out_dir, exist_ok=True)
    ckpt_path = os.path.join(paths.out_dir, ckpt_name)

    df_train_full = build_seg_df(paths.seg_train_img_dir, paths.seg_train_msk_dir)
    df_test = build_seg_df(paths.seg_test_img_dir, paths.seg_test_msk_dir)
    train_df, val_df = stratified_split(df_train_full, val_size=0.2, seed=42)

    train_loader = DataLoader(BriscSegDataset(train_df, img_size, augment=True),
                              batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(BriscSegDataset(val_df, img_size, augment=False),
                            batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(BriscSegDataset(df_test, img_size, augment=False),
                             batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model_type = model_type.lower()
    if model_type == "unet":
        model = UNet(in_ch=1, out_ch=1, base=32).to(DEVICE)
    elif model_type in ("attunet", "attention_unet", "attention"):
        model = AttentionUNet(in_ch=1, out_ch=1, base=32).to(DEVICE)
    else:
        raise ValueError("model_type must be one of: 'unet', 'attunet'")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    best_val_dice = -1.0
    no_improve = 0
    history = {"train_loss": [], "val_loss": [], "train_dice": [], "val_dice": []}

    for epoch in range(1, epochs + 1):
        tr_loss, tr_miou, tr_dice, tr_pix = train_seg_one_epoch(model, train_loader, optimizer)
        va_loss, va_miou, va_dice, va_pix = eval_seg(model, val_loader)
        scheduler.step(va_dice)

        history["train_loss"].append(tr_loss); history["val_loss"].append(va_loss)
        history["train_dice"].append(tr_dice); history["val_dice"].append(va_dice)

        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"[Stage-1:{model_type}] Epoch {epoch:02d}/{epochs} | lr={lr_now:.2e} | "
            f"loss {tr_loss:.4f}/{va_loss:.4f} | dice {tr_dice:.4f}/{va_dice:.4f} | miou {tr_miou:.4f}/{va_miou:.4f}"
        )

        if va_dice > best_val_dice + 1e-4:
            best_val_dice = va_dice
            no_improve = 0
            torch.save(model.state_dict(), ckpt_path)
            print(f"  ✓ Saved best: {ckpt_path} (val_dice={best_val_dice:.4f})")
        else:
            no_improve += 1

        if no_improve >= early_stop_patience:
            print("  Early stopping triggered.")
            break

    # summary (best)
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    tr = eval_seg(model, train_loader)
    va = eval_seg(model, val_loader)
    te = eval_seg(model, test_loader)
    summary = pd.DataFrame(
        [["train", *tr], ["val", *va], ["test", *te]],
        columns=["split", "loss", "mIoU", "dice", "pixel_accuracy"],
    )
    print("\n=== Stage-1 Segmentation Summary (best ckpt) ===")
    print(summary.to_string(index=False))

    # curves
    ep = np.arange(1, len(history["train_loss"]) + 1)
    plt.figure(); plt.plot(ep, history["train_loss"], label="train_loss"); plt.plot(ep, history["val_loss"], label="val_loss")
    plt.title(f"Stage-1 {model_type}: Loss"); plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.show()
    plt.figure(); plt.plot(ep, history["train_dice"], label="train_dice"); plt.plot(ep, history["val_dice"], label="val_dice")
    plt.title(f"Stage-1 {model_type}: Dice"); plt.xlabel("Epoch"); plt.ylabel("Dice"); plt.legend(); plt.show()

    return model, ckpt_path, summary

def run_stage2_classification(paths: Paths,
                              stage1_ckpt_path: str,
                              img_size=(384, 384),
                              batch_size: int = 16,
                              epochs: int = 40,
                              lr: float = 1e-3,
                              ckpt_name: str = "stage2_cls_best.pth"):
    os.makedirs(paths.out_dir, exist_ok=True)
    ckpt_path = os.path.join(paths.out_dir, ckpt_name)

    # build splits from segmentation_task images (same as your original)
    train_img_dir = os.path.join(paths.data_root, "segmentation_task", "train", "images")
    test_img_dir  = os.path.join(paths.data_root, "segmentation_task", "test",  "images")

    df_train_full = build_cls_df(train_img_dir, "train", paths.data_root)
    df_test = build_cls_df(test_img_dir, "test", paths.data_root)
    train_df, val_df = stratified_split(df_train_full, val_size=0.2, seed=42)

    train_loader = DataLoader(BriscClsDatasetGray(train_df, paths.data_root, img_size, augment=True),
                              batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(BriscClsDatasetGray(val_df, paths.data_root, img_size, augment=False),
                            batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(BriscClsDatasetGray(df_test, paths.data_root, img_size, augment=False),
                             batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # Load encoder weights from Stage-1 checkpoint
    encoder = UNetEncoder(in_ch=1, base=32)
    state = torch.load(stage1_ckpt_path, map_location="cpu")
    enc_state = {k: v for k, v in state.items() if k.startswith(("enc1", "enc2", "enc3", "enc4", "bottleneck"))}
    encoder.load_state_dict(enc_state, strict=False)

    # Freeze encoder
    for p in encoder.parameters():
        p.requires_grad = False

    model = EncoderClassifier(encoder, feat_ch=32 * 16, num_classes=len(LABELS)).to(DEVICE)

    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)

    best_val_f1 = -1.0
    history = {"train_loss": [], "val_loss": [], "train_f1": [], "val_f1": []}

    for epoch in range(1, epochs + 1):
        tr = train_cls_one_epoch(model, train_loader, optimizer, num_classes=len(LABELS))
        va = eval_cls(model, val_loader, num_classes=len(LABELS))

        history["train_loss"].append(tr[0]); history["val_loss"].append(va[0])
        history["train_f1"].append(tr[4]); history["val_f1"].append(va[4])

        print(
            f"[Stage-2] Epoch {epoch:02d}/{epochs} | "
            f"loss {tr[0]:.4f}/{va[0]:.4f} | acc {tr[1]:.4f}/{va[1]:.4f} | F1 {tr[4]:.4f}/{va[4]:.4f}"
        )

        if va[4] > best_val_f1:
            best_val_f1 = va[4]
            torch.save(model.state_dict(), ckpt_path)
            print(f"  ✓ Saved best: {ckpt_path} (val_F1={best_val_f1:.4f})")

    # summary
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    tr = eval_cls(model, train_loader, len(LABELS))
    va = eval_cls(model, val_loader, len(LABELS))
    te = eval_cls(model, test_loader, len(LABELS))

    summary = pd.DataFrame(
        [["train", *tr], ["val", *va], ["test", *te]],
        columns=["split", "loss", "accuracy", "precision_macro", "recall_macro", "f1_macro"],
    )
    print("\n=== Stage-2 Classification Summary (best ckpt) ===")
    print(summary.to_string(index=False))

    ep = np.arange(1, len(history["train_loss"]) + 1)
    plt.figure(); plt.plot(ep, history["train_loss"], label="train_loss"); plt.plot(ep, history["val_loss"], label="val_loss")
    plt.title("Stage-2: Loss"); plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.show()
    plt.figure(); plt.plot(ep, history["train_f1"], label="train_F1"); plt.plot(ep, history["val_f1"], label="val_F1")
    plt.title("Stage-2: Macro-F1"); plt.xlabel("Epoch"); plt.ylabel("F1"); plt.legend(); plt.show()

    return model, ckpt_path, summary

def run_joint_multitask(paths: Paths,
                        img_size=(384, 384),
                        batch_size: int = 8,
                        epochs: int = 40,
                        lr: float = 1e-3,
                        lambda_cls: float = 0.5,
                        ckpt_name: str = "multitask_unet_best.pth"):
    os.makedirs(paths.out_dir, exist_ok=True)
    ckpt_path = os.path.join(paths.out_dir, ckpt_name)

    df_train_full = build_seg_df(paths.seg_train_img_dir, paths.seg_train_msk_dir)
    df_test = build_seg_df(paths.seg_test_img_dir, paths.seg_test_msk_dir)
    train_df, val_df = stratified_split(df_train_full, val_size=0.2, seed=42)

    train_loader = DataLoader(BriscMultiTaskDataset(train_df, img_size, augment=True),
                              batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(BriscMultiTaskDataset(val_df, img_size, augment=False),
                            batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(BriscMultiTaskDataset(df_test, img_size, augment=False),
                             batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = MultiTaskUNet(in_ch=1, seg_out_ch=1, num_classes=len(LABELS), base=32).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_score = -1.0  # val_dice + val_f1
    history = {"train_loss": [], "val_loss": [], "val_dice": [], "val_f1": []}

    for epoch in range(1, epochs + 1):
        tr_loss, tr_seg, tr_cls = train_joint_one_epoch(model, train_loader, optimizer, lambda_cls=lambda_cls)
        va_loss, va_seg, va_cls = eval_joint(model, val_loader, lambda_cls=lambda_cls)
        va_miou, va_dice, va_pix = va_seg
        va_acc, va_mp, va_mr, va_f1 = va_cls

        history["train_loss"].append(tr_loss); history["val_loss"].append(va_loss)
        history["val_dice"].append(va_dice); history["val_f1"].append(va_f1)

        print(
            f"[Joint] Epoch {epoch:02d}/{epochs} | "
            f"loss {tr_loss:.4f}/{va_loss:.4f} | "
            f"Seg dice {va_dice:.4f} miou {va_miou:.4f} | "
            f"Cls acc {va_acc:.4f} F1 {va_f1:.4f}"
        )

        score = va_dice + va_f1
        if score > best_score:
            best_score = score
            torch.save(model.state_dict(), ckpt_path)
            print(f"  ✓ Saved best: {ckpt_path} (val_dice+val_f1={best_score:.4f})")

    # summary
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    tr = eval_joint(model, train_loader, lambda_cls=lambda_cls)
    va = eval_joint(model, val_loader, lambda_cls=lambda_cls)
    te = eval_joint(model, test_loader, lambda_cls=lambda_cls)

    def row(name, out):
        loss, seg, cls = out
        miou, dice, pix = seg
        acc, mp, mr, f1 = cls
        return [name, loss, miou, dice, pix, acc, mp, mr, f1]

    summary = pd.DataFrame(
        [row("train", tr), row("val", va), row("test", te)],
        columns=["split", "total_loss", "mIoU", "dice", "pixel_accuracy", "accuracy", "precision_macro", "recall_macro", "f1_macro"],
    )
    print("\n=== Joint Training Summary (best ckpt) ===")
    print(summary.to_string(index=False))

    ep = np.arange(1, len(history["train_loss"]) + 1)
    plt.figure(); plt.plot(ep, history["train_loss"], label="train_loss"); plt.plot(ep, history["val_loss"], label="val_loss")
    plt.title("Joint: Total loss"); plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.show()
    plt.figure(); plt.plot(ep, history["val_dice"], label="val_dice"); plt.plot(ep, history["val_f1"], label="val_f1")
    plt.title("Joint: Val Dice & Val F1"); plt.xlabel("Epoch"); plt.legend(); plt.show()

    return model, ckpt_path, summary

def make_imagenet_model(name: str, num_classes: int = 4) -> nn.Module:
    name = name.lower()
    if name == "mobilenet_v2":
        m = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        in_f = m.classifier[1].in_features
        m.classifier[1] = nn.Linear(in_f, num_classes)
        return m
    if name == "efficientnet_b0":
        m = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_f = m.classifier[1].in_features
        m.classifier[1] = nn.Linear(in_f, num_classes)
        return m
    if name == "densenet121":
        m = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
        in_f = m.classifier.in_features
        m.classifier = nn.Linear(in_f, num_classes)
        return m
    raise ValueError("Unknown model name")

def run_backbone_compare(paths: Paths,
                         models: List[str] = ("mobilenet_v2", "efficientnet_b0", "densenet121"),
                         img_size: int = 224,
                         batch_size: int = 32,
                         epochs: int = 15,
                         lr: float = 3e-4):
    # splits
    train_img_dir = os.path.join(paths.data_root, "segmentation_task", "train", "images")
    test_img_dir  = os.path.join(paths.data_root, "segmentation_task", "test",  "images")
    df_train_full = build_cls_df(train_img_dir, "train", paths.data_root)
    df_test = build_cls_df(test_img_dir, "test", paths.data_root)
    train_df, val_df = stratified_split(df_train_full, val_size=0.2, seed=42)

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD  = (0.229, 0.224, 0.225)

    train_tf = T.Compose([
        T.Resize((img_size, img_size)),
        T.RandomResizedCrop(img_size, scale=(0.85, 1.0), ratio=(0.9, 1.1)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=10),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_tf = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    train_loader = DataLoader(BriscClsDatasetRGB(train_df, paths.data_root, train_tf),
                              batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(BriscClsDatasetRGB(val_df, paths.data_root, eval_tf),
                            batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(BriscClsDatasetRGB(df_test, paths.data_root, eval_tf),
                             batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # local AMP for this loop (uses torch.autocast for cuda)
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE == "cuda"))

    def train_one_epoch_backbone(model, loader, optimizer):
        model.train()
        total_loss, n = 0.0, 0
        cm = np.zeros((len(LABELS), len(LABELS)), dtype=np.int64)

        for x, y in loader:
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type="cuda", enabled=(DEVICE == "cuda")):
                logits = model(x)
                loss = _ce(logits, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += float(loss.item()); n += 1
            preds = logits.argmax(dim=1).detach().cpu().numpy()
            trues = y.detach().cpu().numpy()
            for t, p in zip(trues, preds):
                cm[t, p] += 1

        acc, mp, mr, mf1 = cls_metrics_from_confmat(cm)
        return total_loss / max(1, n), acc, mp, mr, mf1

    @torch.no_grad()
    def eval_backbone(model, loader):
        model.eval()
        total_loss, n = 0.0, 0
        cm = np.zeros((len(LABELS), len(LABELS)), dtype=np.int64)

        for x, y in loader:
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)
            logits = model(x)
            loss = _ce(logits, y)

            total_loss += float(loss.item()); n += 1
            preds = logits.argmax(dim=1).detach().cpu().numpy()
            trues = y.detach().cpu().numpy()
            for t, p in zip(trues, preds):
                cm[t, p] += 1

        acc, mp, mr, mf1 = cls_metrics_from_confmat(cm)
        return total_loss / max(1, n), acc, mp, mr, mf1

    rows = []
    histories = {}

    for name in models:
        print("\n" + "=" * 70)
        print("Backbone:", name)
        print("=" * 70)

        model = make_imagenet_model(name, num_classes=len(LABELS)).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

        best_val_f1 = -1.0
        best_path = os.path.join(paths.out_dir, f"{name}_best.pth")
        hist = {"val_loss": [], "val_f1": []}

        for epoch in range(1, epochs + 1):
            tr = train_one_epoch_backbone(model, train_loader, optimizer)
            va = eval_backbone(model, val_loader)
            scheduler.step(va[4])

            hist["val_loss"].append(va[0]); hist["val_f1"].append(va[4])
            lr_now = optimizer.param_groups[0]["lr"]

            print(
                f"Epoch {epoch:02d}/{epochs} | lr={lr_now:.2e} | "
                f"train_acc={tr[1]:.4f} val_acc={va[1]:.4f} | val_F1={va[4]:.4f}"
            )

            if va[4] > best_val_f1:
                best_val_f1 = va[4]
                torch.save(model.state_dict(), best_path)
                print(f"  ✓ Saved best: {best_path} (val_F1={best_val_f1:.4f})")

        model.load_state_dict(torch.load(best_path, map_location=DEVICE))
        tr = eval_backbone(model, train_loader)
        va = eval_backbone(model, val_loader)
        te = eval_backbone(model, test_loader)

        rows += [
            [name, "train", *tr],
            [name, "val",   *va],
            [name, "test",  *te],
        ]
        histories[name] = hist

    results = pd.DataFrame(rows, columns=["model", "split", "loss", "accuracy", "precision_macro", "recall_macro", "f1_macro"])
    print("\n=== Backbone comparison (train/val/test) ===")
    print(results.to_string(index=False))

    test_only = results[results["split"] == "test"].sort_values("f1_macro", ascending=False)
    print("\n=== TEST ONLY (sorted by F1) ===")
    print(test_only.to_string(index=False))

    # overlay curves
    plt.figure(figsize=(7, 5))
    for m in models:
        plt.plot(histories[m]["val_loss"], label=m)
    plt.title("Validation loss (overlay)"); plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.show()

    plt.figure(figsize=(7, 5))
    for m in models:
        plt.plot(histories[m]["val_f1"], label=m)
    plt.title("Validation Macro-F1 (overlay)"); plt.xlabel("Epoch"); plt.ylabel("F1"); plt.legend(); plt.show()

    return results


# =========================================================
# 9) CONFIG: what to run
# =========================================================
if __name__ == "__main__":
    paths = Paths(
        data_root=os.environ.get("BRISC_DATA_ROOT", "./data/brisc2025"),
        out_dir=os.environ.get("BRISC_OUT_DIR", "./outputs"),
    )
    print("DEVICE:", DEVICE)
    print("DATA_ROOT:", paths.data_root)
    print("OUT_DIR  :", paths.out_dir)

    # --- Toggle these flags ---
    RUN_STAGE1_UNET = False
    RUN_STAGE1_ATTUNET = False
    RUN_STAGE2 = False
    RUN_JOINT = False
    RUN_BACKBONE_COMPARE = False

    # --- Stage-1 segmentation ---
    if RUN_STAGE1_UNET:
        seg_model, seg_ckpt, seg_summary = run_stage1_segmentation(
            paths, model_type="unet", ckpt_name="stage1_unet_seg_best.pth"
        )

    if RUN_STAGE1_ATTUNET:
        seg_model, seg_ckpt, seg_summary = run_stage1_segmentation(
            paths, model_type="attunet", ckpt_name="stage1_attunet_seg_best.pth"
        )

    # --- Stage-2 classification (needs a Stage-1 ckpt) ---
    if RUN_STAGE2:
        stage1_ckpt = os.path.join(paths.out_dir, "stage1_unet_seg_best.pth")
        cls_model, cls_ckpt, cls_summary = run_stage2_classification(
            paths, stage1_ckpt_path=stage1_ckpt, ckpt_name="stage2_cls_best.pth"
        )

    # --- Joint multi-task ---
    if RUN_JOINT:
        joint_model, joint_ckpt, joint_summary = run_joint_multitask(paths, ckpt_name="multitask_unet_best.pth")

    # --- Compare 3 classifier backbones ---
    if RUN_BACKBONE_COMPARE:
        run_backbone_compare(paths)
