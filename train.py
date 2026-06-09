import argparse
import copy
import csv
import json
import math
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from datareader import (
    DEFAULT_LEADS,
    get_ludb_af_afl_label,
    list_ludb_records,
    load_ludb_single_lead_tensors,
    load_ludb_window_tensors,
)
from loss import FocalLoss
from model import ECGUNet3p
from transform import (
    BaselineShift,
    BaselineWander,
    ChannelResize,
    Compose,
    CustomTensorDataset,
    GaussianNoise,
    PowerlineNoise,
)


# ── Device / seed / init ──────────────────────────────────────────────────────

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda:0')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def initialize_he(module):
    if isinstance(module, (torch.nn.Conv1d, torch.nn.Linear)):
        torch.nn.init.kaiming_normal_(module.weight, nonlinearity='leaky_relu')
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)


# ── Record splitting ──────────────────────────────────────────────────────────

def split_records(records, n_train, split_strategy, seed):
    if split_strategy == 'sequential':
        return records[:n_train], records[n_train:]

    rng = random.Random(seed)
    negative_records = [r for r in records if get_ludb_af_afl_label(r) == 0]
    positive_records = [r for r in records if get_ludb_af_afl_label(r) == 1]
    if len(negative_records) < 2 or len(positive_records) < 2:
        return records[:n_train], records[n_train:]

    rng.shuffle(negative_records)
    rng.shuffle(positive_records)

    pos_train = round(n_train * len(positive_records) / len(records))
    pos_train = min(max(1, pos_train), len(positive_records) - 1)
    neg_train = min(max(1, n_train - pos_train), len(negative_records) - 1)

    train_records = negative_records[:neg_train] + positive_records[:pos_train]
    test_records = negative_records[neg_train:] + positive_records[pos_train:]
    return (
        sorted(train_records, key=lambda r: records.index(r)),
        sorted(test_records, key=lambda r: records.index(r)),
    )


# ── Augmentation ──────────────────────────────────────────────────────────────

def make_train_transform(_signal_length, sample_rate):
    return Compose([
        BaselineWander(prob=0.2, freq=sample_rate),
        GaussianNoise(prob=0.2),
        PowerlineNoise(prob=0.2, freq=sample_rate),
        ChannelResize(),
        BaselineShift(prob=0.2),
    ])


# ── Cross-task helpers ────────────────────────────────────────────────────────

def _balanced_accuracy(preds, targets):
    pos_mask = targets == 1
    neg_mask = targets == 0
    recall_pos = (preds[pos_mask] == 1).float().mean().item() if pos_mask.sum() > 0 else None
    recall_neg = (preds[neg_mask] == 0).float().mean().item() if neg_mask.sum() > 0 else None
    if recall_pos is not None and recall_neg is not None:
        return (recall_pos + recall_neg) / 2
    return recall_pos or recall_neg or 0.0


def _binary_cls_stats(preds, targets):
    pos_mask = targets == 1
    neg_mask = targets == 0
    pos_recall = (preds[pos_mask] == 1).float().mean().item() if pos_mask.sum() > 0 else None
    neg_recall = (preds[neg_mask] == 0).float().mean().item() if neg_mask.sum() > 0 else None
    return {
        'pos_recall': pos_recall if pos_recall is not None else 0.0,
        'neg_recall': neg_recall if neg_recall is not None else 0.0,
        'pred_pos_rate': (preds == 1).float().mean().item() if len(preds) else 0.0,
    }


def _preds_from_p_prob(p_probs, threshold):
    return (p_probs >= threshold).long()


def _cls_metrics_from_probs(p_probs, targets, threshold):
    preds = _preds_from_p_prob(p_probs, threshold)
    cls_stats = _binary_cls_stats(preds, targets)
    return {
        'accuracy': (preds == targets).float().mean().item(),
        'balanced_acc': _balanced_accuracy(preds, targets),
        **cls_stats,
    }


def _best_balanced_threshold(p_probs, targets):
    """Choose a P-present probability threshold that maximizes train balanced accuracy."""
    p_probs = p_probs.detach().cpu().float()
    targets = targets.detach().cpu().long()
    candidates = torch.cat([
        torch.tensor([0.0]),
        torch.unique(p_probs),
        torch.tensor([1.0 + 1e-6]),
    ])

    best_threshold = 0.5
    best_balanced_acc = -1.0
    for threshold in candidates:
        metrics = _cls_metrics_from_probs(p_probs, targets, float(threshold))
        balanced_acc = metrics['balanced_acc']
        if balanced_acc > best_balanced_acc:
            best_balanced_acc = balanced_acc
            best_threshold = float(threshold)

    return best_threshold, best_balanced_acc


def _average_precision(p_probs, targets):
    p_probs = p_probs.detach().cpu().float()
    targets = targets.detach().cpu().long()
    n_pos = int((targets == 1).sum().item())
    if n_pos == 0:
        return 0.0
    order = torch.argsort(p_probs, descending=True)
    sorted_targets = targets[order]
    tp_cum = torch.cumsum((sorted_targets == 1).float(), dim=0)
    fp_cum = torch.cumsum((sorted_targets == 0).float(), dim=0)
    precision = tp_cum / (tp_cum + fp_cum).clamp_min(1e-12)
    return float((precision * (sorted_targets == 1).float()).sum().item() / n_pos)


def _binary_stats_extended(p_probs, targets, threshold):
    preds = (p_probs >= threshold).long()
    pos_mask = targets == 1
    neg_mask = targets == 0
    tp = int(((preds == 1) & pos_mask).sum().item())
    tn = int(((preds == 0) & neg_mask).sum().item())
    fp = int(((preds == 1) & neg_mask).sum().item())
    fn = int(((preds == 0) & pos_mask).sum().item())
    pos_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    neg_recall = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = (2 * precision * pos_recall) / (precision + pos_recall) if (precision + pos_recall) > 0 else 0.0
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn) - (fp * fn)) / denom if denom > 0 else 0.0
    return {
        'accuracy': (preds == targets).float().mean().item(),
        'balanced_acc': (pos_recall + neg_recall) / 2,
        'precision': precision,
        'f1': f1,
        'mcc': mcc,
        'pr_auc': _average_precision(p_probs, targets),
        'pos_recall': pos_recall,
        'neg_recall': neg_recall,
        'pred_pos_rate': (preds == 1).float().mean().item(),
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn,
    }


def _top_fraction_mean(values, top_fraction=0.10):
    k = max(1, int(round(values.shape[-1] * top_fraction)))
    return values.topk(k, dim=1).values.mean(dim=1)


def _safe_slice_stats(values):
    return values.max(dim=1).values, _top_fraction_mean(values), values.mean(dim=1)


def _qrs_anchor(length, qrs_anchor_index=None):
    anchor = qrs_anchor_index if qrs_anchor_index is not None else int(round(length * 0.8))
    return max(1, min(int(anchor), length - 1))


def _normalized_entropy(prob_maps):
    eps = 1e-6
    probs = prob_maps.clamp(eps, 1 - eps)
    entropy = -(probs * probs.log()).sum(dim=1)
    return entropy / np.log(prob_maps.shape[1])


def _temperature_scale_prob_maps(prob_maps, temperature):
    if temperature is None:
        return prob_maps
    log_probs = prob_maps.clamp(1e-6, 1 - 1e-6).log()
    return torch.softmax(log_probs / temperature, dim=1)


def _segmenter_window_features_from_probs(probs, qrs_anchor_index=None):
    """Pool segmenter probabilities into beat-level P-present features.

    The classifier sees explicit morphology evidence instead of raw ECG. The
    anchor is QRS onset in the extracted window, so pre-anchor P evidence is
    the most important negative/positive separator.
    """
    p_prob = probs[:, 0, :]
    qrs_prob = probs[:, 1, :]
    t_prob = probs[:, 2, :]
    none_prob = probs[:, 3, :]
    length = p_prob.shape[-1]

    anchor = _qrs_anchor(length, qrs_anchor_index=qrs_anchor_index)
    p_pre = p_prob[:, :anchor]
    p_post = p_prob[:, anchor:]

    p_max, p_top, p_mean = _safe_slice_stats(p_prob)
    p_pre_max, p_pre_top, p_pre_mean = _safe_slice_stats(p_pre)
    p_post_max, p_post_top, p_post_mean = _safe_slice_stats(p_post)

    return torch.stack([
        p_max,
        p_top,
        p_mean,
        p_pre_max,
        p_pre_top,
        p_pre_mean,
        p_post_max,
        p_post_top,
        p_post_mean,
        qrs_prob.max(dim=1).values,
        qrs_prob.mean(dim=1),
        t_prob.mean(dim=1),
        none_prob.mean(dim=1),
    ], dim=1)


def _tokenize_prob_maps(prob_maps, qrs_anchor_index, n_pre_bins=4, n_post_bins=2):
    """Structured token features from probability maps [B, C, T].

    The window is split into pre-QRS and post-QRS bins to preserve where
    evidence appears. Each bin contributes mean and max probability per class.
    """
    length = prob_maps.shape[-1]
    anchor = _qrs_anchor(length, qrs_anchor_index=qrs_anchor_index)

    spans = []
    pre_edges = torch.linspace(0, anchor, steps=n_pre_bins + 1, dtype=torch.long)
    post_edges = torch.linspace(anchor, length, steps=n_post_bins + 1, dtype=torch.long)

    for start, end in zip(pre_edges[:-1], pre_edges[1:]):
        spans.append((int(start.item()), int(end.item())))
    for start, end in zip(post_edges[:-1], post_edges[1:]):
        spans.append((int(start.item()), int(end.item())))

    tokens = []
    for start, end in spans:
        end = max(end, start + 1)
        slice_probs = prob_maps[:, :, start:end]
        tokens.append(torch.cat([
            slice_probs.mean(dim=-1),
            slice_probs.max(dim=-1).values,
        ], dim=1))
    return torch.cat(tokens, dim=1)


def _uncertainty_context_features(prob_maps, qrs_anchor_index):
    entropy = _normalized_entropy(prob_maps)
    p_prob = prob_maps[:, 0, :]
    qrs_prob = prob_maps[:, 1, :]
    anchor = _qrs_anchor(prob_maps.shape[-1], qrs_anchor_index=qrs_anchor_index)
    entropy_pre = entropy[:, :anchor]
    entropy_post = entropy[:, anchor:]
    p_pre = p_prob[:, :anchor]
    qrs_post = qrs_prob[:, anchor:]

    return torch.stack([
        entropy.mean(dim=1),
        entropy_pre.mean(dim=1),
        _top_fraction_mean(p_pre),
        qrs_post.mean(dim=1),
    ], dim=1)


def _noisy_or_p_score(prob_maps, qrs_anchor_index):
    anchor = _qrs_anchor(prob_maps.shape[-1], qrs_anchor_index=qrs_anchor_index)
    p_pre = prob_maps[:, 0, :anchor].clamp(1e-6, 1 - 1e-6)
    return 1 - torch.prod(1 - p_pre, dim=1)


def _calibrated_teacher_features(prob_maps, qrs_anchor_index, teacher_temperature):
    calibrated = _temperature_scale_prob_maps(prob_maps, teacher_temperature)
    raw_score = _noisy_or_p_score(prob_maps, qrs_anchor_index=qrs_anchor_index)
    cal_score = _noisy_or_p_score(calibrated, qrs_anchor_index=qrs_anchor_index)
    anchor = _qrs_anchor(prob_maps.shape[-1], qrs_anchor_index=qrs_anchor_index)
    cal_entropy = _normalized_entropy(calibrated)[:, :anchor].mean(dim=1)
    return torch.stack([
        cal_score,
        (cal_score - 0.5).abs(),
        raw_score,
        cal_score - raw_score,
        cal_entropy,
    ], dim=1)


def _context_feature_vector(prob_maps, beat_metadata, qrs_anchor_index, teacher_temperature=None):
    if beat_metadata is None:
        raise ValueError('Context feature vector requires beat_metadata.')
    features = [
        beat_metadata,
        _uncertainty_context_features(prob_maps, qrs_anchor_index=qrs_anchor_index),
    ]
    if teacher_temperature is not None:
        features.append(_calibrated_teacher_features(
            prob_maps,
            qrs_anchor_index=qrs_anchor_index,
            teacher_temperature=teacher_temperature,
        ))
    return torch.cat(features, dim=1)


def fit_teacher_temperature(prob_maps, labels, qrs_anchor_index):
    device = prob_maps.device
    labels = labels.to(device).float()
    log_temperature = torch.nn.Parameter(torch.zeros(1, device=device))
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=50, line_search_fn='strong_wolfe')

    def closure():
        optimizer.zero_grad()
        temperature = log_temperature.exp().clamp(1e-3, 100.0)
        calibrated = _temperature_scale_prob_maps(prob_maps, temperature)
        q = _noisy_or_p_score(calibrated, qrs_anchor_index=qrs_anchor_index).clamp(1e-6, 1 - 1e-6)
        loss = torch.nn.functional.binary_cross_entropy(q, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temperature.exp().detach().cpu().item())


def _classifier_features(prob_maps, classifier_input, qrs_anchor_index, beat_metadata=None, teacher_temperature=None):
    if classifier_input in {'seg_features', 'oracle_features'}:
        return _segmenter_window_features_from_probs(prob_maps, qrs_anchor_index=qrs_anchor_index)
    if classifier_input in {'seg_tokens', 'oracle_tokens'}:
        return _tokenize_prob_maps(prob_maps, qrs_anchor_index=qrs_anchor_index)
    if classifier_input in {'seg_tokens_ctx', 'oracle_tokens_ctx'}:
        if beat_metadata is None:
            raise ValueError(f'{classifier_input} requires beat_metadata.')
        return torch.cat([
            _tokenize_prob_maps(prob_maps, qrs_anchor_index=qrs_anchor_index),
            beat_metadata,
            _uncertainty_context_features(prob_maps, qrs_anchor_index=qrs_anchor_index),
        ], dim=1)
    if classifier_input in {'seg_tokens_ctx_cal', 'oracle_tokens_ctx_cal'}:
        if beat_metadata is None:
            raise ValueError(f'{classifier_input} requires beat_metadata.')
        return torch.cat([
            _tokenize_prob_maps(prob_maps, qrs_anchor_index=qrs_anchor_index),
            _context_feature_vector(
                prob_maps,
                beat_metadata=beat_metadata,
                qrs_anchor_index=qrs_anchor_index,
                teacher_temperature=teacher_temperature,
            ),
        ], dim=1)
    raise ValueError(f'Unsupported classifier_input={classifier_input!r}')


# ── Epoch runners ─────────────────────────────────────────────────────────────

def run_seg_epoch(seg_model, device, loader, loss_fn_seg, optimizer=None):
    """Segmenter epoch.  Pass optimizer=None for eval (no grad)."""
    is_training = optimizer is not None
    seg_model.train(is_training)
    total_loss_seg = 0.0
    n = len(loader.dataset)

    ctx = torch.enable_grad() if is_training else torch.no_grad()
    with ctx:
        for data, target, _ in tqdm(loader, desc='seg', leave=False):
            data, target = data.to(device), target.to(device)
            logits = seg_model(data)
            loss = loss_fn_seg(logits, torch.argmax(target, dim=1))
            if is_training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss_seg += loss.item() * len(data)

    return {'loss_seg': total_loss_seg / n}


# ── Main training function ────────────────────────────────────────────────────

def train_base_segmenter(
    n_channels,
    epochs,
    batch_size,
    seg_lr,
    focal_gamma,
    data_dir,
    output_dir,
    seed,
    n_ludb_train,
    target_fs,
    source_fs,
    split_strategy,
    early_stopping_epochs,
    early_stopping_min_delta,
):
    set_seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    records = list_ludb_records(data_dir)
    train_records, test_records = split_records(records, n_ludb_train, split_strategy, seed)

    print('Loading full-signal tensors for segmenter...')
    X_train, y_train, sample_rate = load_ludb_single_lead_tensors(
        train_records, leads=DEFAULT_LEADS, target_fs=target_fs, source_fs=source_fs,
    )
    X_test, y_test, _ = load_ludb_single_lead_tensors(
        test_records, leads=DEFAULT_LEADS, target_fs=sample_rate, source_fs=source_fs,
    )

    print(f'Sample rate: {sample_rate:g} Hz')
    print(f'Seg  train: {tuple(X_train.shape)},  test: {tuple(X_test.shape)}')

    device = get_device()
    print(f'Device: {device}')

    seg_model = ECGUNet3p(n_channels=n_channels).to(device)
    seg_model.apply(initialize_he)
    print(f'ECGUNet3p: {sum(p.numel() for p in seg_model.parameters()):,} params')

    loss_fn_seg = FocalLoss(gamma=focal_gamma)
    seg_opt = torch.optim.Adam(seg_model.parameters(), lr=seg_lr)
    seg_sch = torch.optim.lr_scheduler.CosineAnnealingLR(seg_opt, T_max=epochs, eta_min=1e-5)

    seg_train_ds = CustomTensorDataset(
        tensors=(X_train, y_train, torch.zeros(len(X_train), dtype=torch.long)),
        transform=make_train_transform(X_train.shape[-1], sample_rate),
    )
    seg_test_ds = CustomTensorDataset(
        tensors=(X_test, y_test, torch.zeros(len(X_test), dtype=torch.long)),
    )
    seg_train_loader = DataLoader(seg_train_ds, batch_size=batch_size, shuffle=True)
    seg_test_loader = DataLoader(seg_test_ds, batch_size=batch_size, shuffle=False)

    metrics_path = os.path.join(output_dir, 'metrics.csv')
    fieldnames = ['epoch', 'train_seg_loss', 'test_seg_loss', 'seg_lr']
    with open(metrics_path, 'w', newline='') as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    best_seg_loss = float('inf')
    best_epoch = 0
    es_counter = 0

    def _save(name, epoch, test_seg_loss):
        payload = {
            'epoch': epoch,
            'seg_model_state_dict': seg_model.state_dict(),
            'args': {
                'model_name': 'ECGUNet3p',
                'n_channels': n_channels,
                'batch_size': batch_size,
                'seg_lr': seg_lr,
                'focal_gamma': focal_gamma,
                'seed': seed,
                'n_ludb_train': n_ludb_train,
                'sample_rate': sample_rate,
                'source_fs': source_fs,
                'leads': DEFAULT_LEADS,
                'split_strategy': split_strategy,
                'train_records': [os.path.basename(record) for record in train_records],
                'test_records': [os.path.basename(record) for record in test_records],
            },
            'summary': {
                'best_epoch': epoch,
                'test_seg_loss': test_seg_loss,
            },
        }
        torch.save(payload, os.path.join(output_dir, name))
        print(f'  -> {name}')

    for epoch in range(1, epochs + 1):
        seg_train = run_seg_epoch(seg_model, device, seg_train_loader, loss_fn_seg, seg_opt)
        seg_test = run_seg_epoch(seg_model, device, seg_test_loader, loss_fn_seg)
        seg_sch.step()

        row = {
            'epoch': epoch,
            'train_seg_loss': f'{seg_train["loss_seg"]:.6f}',
            'test_seg_loss': f'{seg_test["loss_seg"]:.6f}',
            'seg_lr': f'{seg_sch.get_last_lr()[0]:.2e}',
        }
        with open(metrics_path, 'a', newline='') as f:
            csv.DictWriter(f, fieldnames=fieldnames).writerow(row)

        if epoch % 10 == 0 or epoch == 1:
            print(
                f'Epoch {epoch:4d}/{epochs}  '
                f'seg train/test={float(row["train_seg_loss"]):.4f}/{float(row["test_seg_loss"]):.4f}  '
                f'lr seg={row["seg_lr"]}'
            )

        if seg_test['loss_seg'] < best_seg_loss - early_stopping_min_delta:
            best_seg_loss = seg_test['loss_seg']
            best_epoch = epoch
            _save('best_seg_model.pt', epoch, best_seg_loss)
            es_counter = 0
        else:
            es_counter += 1

        if early_stopping_epochs > 0 and es_counter >= early_stopping_epochs:
            print(f'Early stopping at epoch {epoch}.')
            break

    print(f'Done. Metrics -> {metrics_path}')

def _soft_p_pre_dice(prob_maps, seg_targets, qrs_anchor_index):
    gt = seg_targets[:, 0, :qrs_anchor_index]
    pred = prob_maps[:, 0, :qrs_anchor_index]
    intersection = (pred * gt).sum(dim=1)
    denom = pred.sum(dim=1) + gt.sum(dim=1)
    return (2 * intersection) / denom.clamp_min(1e-6)


def _absent_p_pre_penalty(prob_maps, p_labels, qrs_anchor_index):
    p_pre = prob_maps[:, 0, :qrs_anchor_index].mean(dim=1)
    absent_mask = p_labels == 0
    if absent_mask.any():
        return p_pre[absent_mask].mean()
    return p_pre.mean() * 0.0


def _evaluate_teacher(seg_model, loader, qrs_anchor_index, device):
    seg_model.eval()
    probs_all = []
    labels_all = []
    dices_all = []
    with torch.no_grad():
        for windows, p_labels, seg_targets in loader:
            windows = windows.to(device)
            seg_targets = seg_targets.to(device)
            probs = torch.softmax(seg_model(windows), dim=1)
            probs_all.append(probs.cpu())
            labels_all.append(p_labels.cpu())
            dices_all.append(_soft_p_pre_dice(probs, seg_targets, qrs_anchor_index).cpu())
    return torch.cat(probs_all), torch.cat(labels_all), torch.cat(dices_all)


def _binary_metrics(y_true, probs, threshold):
    preds = (probs >= threshold).long()
    pos_mask = y_true == 1
    neg_mask = y_true == 0
    pos_recall = (preds[pos_mask] == 1).float().mean().item() if pos_mask.any() else 0.0
    neg_recall = (preds[neg_mask] == 0).float().mean().item() if neg_mask.any() else 0.0
    return {
        'accuracy': (preds == y_true).float().mean().item(),
        'balanced_acc': (pos_recall + neg_recall) / 2,
        'pos_recall': pos_recall,
        'neg_recall': neg_recall,
        'pred_pos_rate': (preds == 1).float().mean().item(),
    }


def refine_segmenter(
    data_dir,
    seg_checkpoint,
    output_dir,
    seed,
    epochs,
    batch_size,
    lr,
    lambda_p_pre,
    lambda_p_absent,
    target_fs,
    source_fs,
    n_ludb_train,
    split_strategy,
    window_pre_ms,
    window_post_ms,
    p_min_overlap_ms,
    p_label_post_ms,
):
    os.makedirs(output_dir, exist_ok=True)
    set_seed(seed)

    records = list_ludb_records(data_dir)
    train_records, test_records = split_records(records, n_ludb_train, split_strategy, seed)
    qrs_anchor_index = int(round(window_pre_ms * target_fs / 1000.0))

    X_train, y_train, y_seg_train, _, _ = load_ludb_window_tensors(
        train_records,
        leads=DEFAULT_LEADS,
        target_fs=target_fs,
        source_fs=source_fs,
        window_pre_ms=window_pre_ms,
        window_post_ms=window_post_ms,
        p_min_overlap_ms=p_min_overlap_ms,
        p_label_post_ms=p_label_post_ms,
        return_seg_targets=True,
    )
    X_test, y_test, y_seg_test, _, _ = load_ludb_window_tensors(
        test_records,
        leads=DEFAULT_LEADS,
        target_fs=target_fs,
        source_fs=source_fs,
        window_pre_ms=window_pre_ms,
        window_post_ms=window_post_ms,
        p_min_overlap_ms=p_min_overlap_ms,
        p_label_post_ms=p_label_post_ms,
        return_seg_targets=True,
    )

    train_loader = DataLoader(TensorDataset(X_train, y_train, y_seg_train), batch_size=batch_size, shuffle=True)
    train_eval_loader = DataLoader(TensorDataset(X_train, y_train, y_seg_train), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test, y_test, y_seg_test), batch_size=batch_size, shuffle=False)

    device = get_device()
    payload = torch.load(seg_checkpoint, map_location='cpu')
    seg_model = ECGUNet3p(n_channels=payload['args'].get('n_channels', 32)).to(device)
    seg_model.load_state_dict(payload['seg_model_state_dict'])

    optimizer = torch.optim.Adam(seg_model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    rows = []
    best = None
    best_state_dict = None
    for epoch in range(1, epochs + 1):
        seg_model.train()
        total_loss = 0.0
        for windows, p_labels, seg_targets in train_loader:
            windows = windows.to(device)
            p_labels = p_labels.to(device)
            seg_targets = seg_targets.to(device)
            logits = seg_model(windows)
            ce = torch.nn.functional.cross_entropy(logits, seg_targets.argmax(dim=1))
            probs = torch.softmax(logits, dim=1)
            p_pre_loss = 1.0 - _soft_p_pre_dice(probs, seg_targets, qrs_anchor_index).mean()
            p_absent_loss = _absent_p_pre_penalty(probs, p_labels, qrs_anchor_index)
            loss = ce + lambda_p_pre * p_pre_loss + lambda_p_absent * p_absent_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(windows)

        scheduler.step()

        train_prob_maps, train_labels, _ = _evaluate_teacher(seg_model, train_eval_loader, qrs_anchor_index, device)
        test_prob_maps, test_labels, test_soft_dice = _evaluate_teacher(seg_model, test_loader, qrs_anchor_index, device)
        temperature = fit_teacher_temperature(train_prob_maps.to(device), train_labels.to(device), qrs_anchor_index)
        train_cal = _temperature_scale_prob_maps(train_prob_maps.to(device), temperature).cpu()
        test_cal = _temperature_scale_prob_maps(test_prob_maps.to(device), temperature).cpu()
        train_teacher = _noisy_or_p_score(train_cal, qrs_anchor_index).cpu()
        test_teacher = _noisy_or_p_score(test_cal, qrs_anchor_index).cpu()
        threshold, _ = _best_balanced_threshold(train_teacher, train_labels)
        test_metrics = _binary_metrics(test_labels, test_teacher, float(threshold))

        row = {
            'epoch': epoch,
            'train_loss': total_loss / len(train_loader.dataset),
            'teacher_temperature': temperature,
            'teacher_threshold': float(threshold),
            'test_teacher_bal_acc': test_metrics['balanced_acc'],
            'test_teacher_pos_recall': test_metrics['pos_recall'],
            'test_teacher_neg_recall': test_metrics['neg_recall'],
            'test_teacher_pred_pos_rate': test_metrics['pred_pos_rate'],
            'test_positive_soft_p_dice': float(test_soft_dice[test_labels == 1].mean().item()),
        }
        rows.append(row)
        if best is None or row['test_teacher_bal_acc'] > best['test_teacher_bal_acc']:
            best = dict(row)
            best_state_dict = copy.deepcopy(seg_model.state_dict())

    with open(os.path.join(output_dir, 'metrics.csv'), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    checkpoint = {
        'seg_model_state_dict': best_state_dict if best_state_dict is not None else seg_model.state_dict(),
        'args': payload.get('args', {}),
        'fine_tune_args': {
            'seed': seed,
            'epochs': epochs,
            'batch_size': batch_size,
            'lr': lr,
            'lambda_p_pre': lambda_p_pre,
            'lambda_p_absent': lambda_p_absent,
            'target_fs': target_fs,
            'source_fs': source_fs,
            'n_ludb_train': n_ludb_train,
            'split_strategy': split_strategy,
            'window_pre_ms': window_pre_ms,
            'window_post_ms': window_post_ms,
            'p_min_overlap_ms': p_min_overlap_ms,
            'p_label_post_ms': p_label_post_ms,
        },
        'summary': best,
    }
    torch.save(checkpoint, os.path.join(output_dir, 'best_seg_model.pt'))
    with open(os.path.join(output_dir, 'summary.json'), 'w') as f:
        json.dump(best, f, indent=2)
    print(json.dumps(best, indent=2))


def _parse_cli_args():
    parser = argparse.ArgumentParser(
        description='Train ECG models for the kept LUDB -> PTB-XL workflow.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest='stage', required=True)

    base = subparsers.add_parser('base', help='Train the base LUDB segmenter.')
    base.add_argument('--data_dir', type=str, required=True)
    base.add_argument('--output_dir', type=str, required=True)
    base.add_argument('--seed', type=int, default=42)
    base.add_argument('--n_ludb_train', type=int, default=100)
    base.add_argument('--target-fs', type=float, default=500)
    base.add_argument('--source-fs', type=float, default=None)
    base.add_argument('--split-strategy', choices=['stratified', 'sequential'], default='stratified')
    base.add_argument('--epochs', type=int, default=1000)
    base.add_argument('--batch-size', type=int, default=32)
    base.add_argument('-c', '--n_channels', type=int, default=32)
    base.add_argument('--focal-gamma', type=float, default=1.0)
    base.add_argument('--seg-lr', type=float, default=1e-3)
    base.add_argument('--early-stopping-epochs', type=int, default=200)
    base.add_argument('--early-stopping-min-delta', type=float, default=0.0)

    refine = subparsers.add_parser('refine', help='Refine the kept segmenter with the P-sensitive loss.')
    refine.add_argument('--data_dir', type=str, required=True)
    refine.add_argument('--seg-checkpoint', type=str, required=True)
    refine.add_argument('--output_dir', type=str, required=True)
    refine.add_argument('--seed', type=int, default=42)
    refine.add_argument('--epochs', type=int, default=15)
    refine.add_argument('--batch-size', type=int, default=128)
    refine.add_argument('--lr', type=float, default=1e-4)
    refine.add_argument('--lambda-p-pre', type=float, default=1.0)
    refine.add_argument('--lambda-p-absent', type=float, default=1.0)
    refine.add_argument('--target-fs', type=float, default=500)
    refine.add_argument('--source-fs', type=float, default=None)
    refine.add_argument('--n_ludb_train', type=int, default=100)
    refine.add_argument('--split-strategy', choices=['stratified', 'sequential'], default='stratified')
    refine.add_argument('--window-pre-ms', type=float, default=300)
    refine.add_argument('--window-post-ms', type=float, default=80)
    refine.add_argument('--p-min-overlap-ms', type=float, default=20)
    refine.add_argument('--p-label-post-ms', type=float, default=40)
    return parser.parse_args()


def main():
    args = _parse_cli_args()
    if args.stage == 'base':
        train_base_segmenter(
            n_channels=args.n_channels,
            epochs=args.epochs,
            batch_size=args.batch_size,
            seg_lr=args.seg_lr,
            focal_gamma=args.focal_gamma,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            seed=args.seed,
            n_ludb_train=args.n_ludb_train,
            target_fs=args.target_fs,
            source_fs=args.source_fs,
            split_strategy=args.split_strategy,
            early_stopping_epochs=args.early_stopping_epochs,
            early_stopping_min_delta=args.early_stopping_min_delta,
        )
    else:
        refine_segmenter(
            data_dir=args.data_dir,
            seg_checkpoint=args.seg_checkpoint,
            output_dir=args.output_dir,
            seed=args.seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            lambda_p_pre=args.lambda_p_pre,
            lambda_p_absent=args.lambda_p_absent,
            target_fs=args.target_fs,
            source_fs=args.source_fs,
            n_ludb_train=args.n_ludb_train,
            split_strategy=args.split_strategy,
            window_pre_ms=args.window_pre_ms,
            window_post_ms=args.window_post_ms,
            p_min_overlap_ms=args.p_min_overlap_ms,
            p_label_post_ms=args.p_label_post_ms,
        )


if __name__ == '__main__':
    main()
