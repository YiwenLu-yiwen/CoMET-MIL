#!/usr/bin/env python3
import argparse
import ast
import csv
import os
import sys
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from datareader import DEFAULT_LEADS, normalize_waveform, resample_waveform, resolve_source_fs
from model import ECGRecordBagMILClassifier, ECGSegmenterFeatureClassifier, ECGUNet3p
from train import (
    _best_balanced_threshold,
    _binary_stats_extended,
    _classifier_features,
    _normalized_entropy,
    get_device,
    initialize_he,
    set_seed,
)

import wfdb


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Record-level AF/AFL inference from segmenter-derived evidence. '
            'Use both --seg-checkpoint and --cls-checkpoint for final inference; '
            'omit --cls-checkpoint only when training a new downstream classifier.'
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument(
        '--seg-checkpoint',
        type=str,
        required=True,
        help='Path to the upstream segmenter checkpoint used to produce evidence features.',
    )
    parser.add_argument(
        '--cls-checkpoint',
        type=str,
        default=None,
        help='Path to the downstream classifier checkpoint. If omitted, a new classifier is trained.',
    )
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument(
        '--modes',
        nargs='+',
        choices=['record_mean_ctx_cal', 'record_mil_ctx_cal'],
        required=True,
    )
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--feature-batch-size', type=int, default=128)
    parser.add_argument('--cls-lr', type=float, default=5e-4)
    parser.add_argument('--target-fs', type=float, default=500)
    parser.add_argument(
        '--source-fs',
        type=float,
        default=None,
        help='Fallback source sample rate used only when the waveform files do not expose record.fs.',
    )
    parser.add_argument(
        '--ptbxl-resolution',
        choices=['100', '500'],
        default='100',
        help='Which PTB-XL waveform files to use: records100/filename_lr or records500/filename_hr.',
    )
    parser.add_argument(
        '--restrict-to-available-resolution',
        choices=['100', '500'],
        default=None,
        help='Optionally restrict all splits to records that are available at the specified PTB-XL resolution.',
    )
    parser.add_argument('--lead', type=str, default='ii', choices=DEFAULT_LEADS)
    parser.add_argument('--window-pre-ms', type=float, default=300)
    parser.add_argument('--window-post-ms', type=float, default=80)
    parser.add_argument('--min-qrs-ms', type=float, default=30.0)
    parser.add_argument('--pos-codes', nargs='+', default=['AFIB', 'AFLT'])
    parser.add_argument('--neg-codes', nargs='+', default=['SR'])
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44])
    parser.add_argument('--save-best-cls', action='store_true')
    parser.add_argument(
        '--plot-limit',
        type=int,
        default=0,
        help='Number of record-level seg+cls visualizations to save per split; -1 saves all, 0 disables plots.',
    )
    return parser.parse_args()


def _load_ptbxl_rows(data_dir, pos_codes, neg_codes):
    metadata_path = os.path.join(data_dir, 'ptbxl_database.csv')
    rows = []
    with open(metadata_path, newline='') as f:
        for row in csv.DictReader(f):
            scp_codes = ast.literal_eval(row['scp_codes'])
            has_pos = any(code in scp_codes for code in pos_codes)
            has_neg = any(code in scp_codes for code in neg_codes)
            if has_pos:
                label = 1
            elif has_neg:
                label = 0
            else:
                continue
            rows.append({
                'ecg_id': int(row['ecg_id']),
                'filename_lr': row['filename_lr'],
                'filename_hr': row['filename_hr'],
                'label': label,
                'strat_fold': int(row['strat_fold']),
            })
    return rows


def _split_ptbxl(rows):
    train_rows = [row for row in rows if row['strat_fold'] <= 8]
    val_rows = [row for row in rows if row['strat_fold'] == 9]
    test_rows = [row for row in rows if row['strat_fold'] == 10]
    return train_rows, val_rows, test_rows


def _filter_rows_by_available_resolution(rows, data_dir, resolution):
    if resolution is None:
        return rows
    key = 'filename_hr' if resolution == '500' else 'filename_lr'
    filtered = []
    for row in rows:
        record_path = os.path.join(data_dir, row[key])
        if os.path.exists(record_path + '.hea') and os.path.exists(record_path + '.dat'):
            filtered.append(row)
    return filtered


def _contiguous_segments(labels, class_id, min_len=1):
    segments = []
    start = None
    for idx, label in enumerate(labels):
        if label == class_id and start is None:
            start = idx
        elif label != class_id and start is not None:
            if idx - start >= min_len:
                segments.append((start, idx - 1))
            start = None
    if start is not None and len(labels) - start >= min_len:
        segments.append((start, len(labels) - 1))
    return segments


def _format_binary_label(label):
    return 'AFIB/AFLT' if int(label) == 1 else 'SR'


def _sample_name(row):
    record_stub = os.path.basename(row.get('record_path', '') or f"ecg_{row['ecg_id']}")
    record_stub = record_stub.replace(os.sep, '_')
    return f"{row['ecg_id']}_{record_stub}"


def _serialize_positions(values):
    if not values:
        return ''
    return ';'.join(str(int(value)) for value in values)


def _plot_record_visualization(
    wave,
    prob_map,
    anchor_samples,
    window_starts,
    window_ends,
    sample_rate,
    row,
    output_path,
    pred_prob,
    pred_label,
    attention=None,
):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    labels = torch.argmax(prob_map, dim=0).cpu().numpy()
    probs = prob_map.cpu().numpy()
    time_axis = np.arange(len(wave)) / sample_rate
    colors = {0: '#d1495b', 1: '#2f9e44', 2: '#5f3dc4'}
    class_labels = {0: 'P', 1: 'QRS', 2: 'T'}
    n_rows = 3 if attention is not None else 2
    fig, axes = plt.subplots(
        n_rows, 1,
        figsize=(18, 9 if attention is not None else 7),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={'height_ratios': [3, 1.4, 1] if attention is not None else [3, 1.4]},
    )
    if n_rows == 2:
        axes = [axes[0], axes[1]]

    ax_wave = axes[0]
    ax_wave.plot(time_axis, wave, color='black', linewidth=0.8, zorder=3)
    for class_id, color in colors.items():
        for start, end in _contiguous_segments(labels, class_id, min_len=1):
            ax_wave.axvspan(start / sample_rate, end / sample_rate, color=color, alpha=0.20, linewidth=0, zorder=1)
    for start, end in zip(window_starts, window_ends):
        ax_wave.axvspan(start / sample_rate, end / sample_rate, color='#74c0fc', alpha=0.08, linewidth=0, zorder=0)
    for beat_index, anchor in enumerate(anchor_samples):
        ax_wave.axvline(anchor / sample_rate, color='#1c7ed6', linestyle='--', linewidth=0.7, alpha=0.7, zorder=2)
        if attention is not None and beat_index < len(attention):
            ax_wave.text(
                anchor / sample_rate,
                ax_wave.get_ylim()[1],
                f'a={attention[beat_index]:.2f}',
                ha='center',
                va='top',
                fontsize=6,
                color='#0b7285',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.65, linewidth=0),
            )
    handles = [Patch(facecolor=colors[c], alpha=0.4, label=class_labels[c]) for c in colors]
    handles.append(Patch(facecolor='#74c0fc', alpha=0.2, label='Beat window'))
    ax_wave.legend(handles=handles, loc='upper right', ncol=4, fontsize='small')
    ax_wave.set_ylabel('Amplitude')
    ax_wave.grid(alpha=0.2)
    ax_wave.set_title(
        f"{_sample_name(row)} | true={_format_binary_label(row['label'])} "
        f"| pred={_format_binary_label(pred_label)} ({pred_prob:.3f})"
    )

    ax_prob = axes[1]
    for class_id, color in colors.items():
        ax_prob.plot(time_axis, probs[class_id], color=color, linewidth=0.9, label=f"{class_labels[class_id]} prob")
    ax_prob.set_ylim(-0.05, 1.05)
    ax_prob.set_ylabel('Probability')
    ax_prob.grid(alpha=0.2)
    ax_prob.legend(loc='upper right', ncol=3, fontsize='small')

    if attention is not None:
        ax_attn = axes[2]
        beat_times = np.asarray(anchor_samples, dtype=np.float64) / sample_rate
        ax_attn.bar(beat_times, attention, width=max(0.02, 0.18), color='#1c7ed6', alpha=0.8)
        ax_attn.set_ylim(0.0, max(1.0, float(np.max(attention)) + 0.1))
        ax_attn.set_ylabel('Attention')
        ax_attn.set_xlabel('Time (s)')
        ax_attn.grid(alpha=0.2)
    else:
        ax_prob.set_xlabel('Time (s)')

    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _extract_record_wave(record_path, lead, target_fs, source_fs=None):
    record = wfdb.rdrecord(record_path)
    lead_to_wave = dict(zip([name.lower() for name in record.sig_name], record.p_signal.T))
    wave = lead_to_wave[lead]
    resolved_source_fs = resolve_source_fs(record, fallback_source_fs=source_fs, record_name=record_path)
    wave = resample_waveform(wave, resolved_source_fs, target_fs)
    wave = normalize_waveform(wave)
    return torch.tensor(wave, dtype=torch.float32).unsqueeze(0).unsqueeze(0)


def _segment_records(seg_model, record_rows, data_dir, lead, target_fs, source_fs, batch_size, device, ptbxl_resolution):
    waves = []
    kept_rows = []
    for row in record_rows:
        rel_path = row['filename_hr'] if ptbxl_resolution == '500' else row['filename_lr']
        record_path = os.path.join(data_dir, rel_path)
        if not os.path.exists(record_path + '.hea') or not os.path.exists(record_path + '.dat'):
            continue
        try:
            waves.append(_extract_record_wave(record_path, lead, target_fs, source_fs=source_fs))
            kept = dict(row)
            kept['record_path'] = rel_path
            kept_rows.append(kept)
        except Exception:
            # Skip files that are still being written or are otherwise unreadable.
            continue
    if not waves:
        raise RuntimeError(f'No PTB-XL waveform files available for lead={lead!r} in the requested split.')
    X = torch.cat(waves, dim=0)
    loader = DataLoader(TensorDataset(X), batch_size=batch_size, shuffle=False)
    probs = []
    seg_model.eval()
    with torch.no_grad():
        for (batch_x,) in loader:
            batch_x = batch_x.to(device)
            probs.append(torch.softmax(seg_model(batch_x), dim=1).cpu())
    return X, torch.cat(probs, dim=0), kept_rows


def _record_instances(prob_maps, target_fs, teacher_temperature, lead, rows, window_pre_ms, window_post_ms, min_qrs_ms):
    pre = int(round(window_pre_ms * target_fs / 1000.0))
    post = int(round(window_post_ms * target_fs / 1000.0))
    min_qrs = max(1, int(round(min_qrs_ms * target_fs / 1000.0)))
    window_len = pre + post
    all_instances = []
    all_labels = []
    stats_rows = []
    plot_rows = []
    feature_dim = None
    max_beats = 1

    for idx, row in enumerate(rows):
        labels = torch.argmax(prob_maps[idx], dim=0).cpu().numpy()
        qrs_segments = _contiguous_segments(labels, class_id=1, min_len=min_qrs)
        anchors = [start for start, _ in qrs_segments]
        if not anchors:
            qrs_prob = prob_maps[idx, 1]
            anchors = [int(torch.argmax(qrs_prob).item())]

        rr_metadata = []
        for beat_idx, anchor in enumerate(anchors):
            prev_rr = ((anchor - anchors[beat_idx - 1]) / target_fs) if beat_idx > 0 else 0.0
            next_rr = ((anchors[beat_idx + 1] - anchor) / target_fs) if beat_idx + 1 < len(anchors) else 0.0
            rr_values = [value for value in (prev_rr, next_rr) if value > 0]
            rr_mean = float(np.mean(rr_values)) if rr_values else 0.0
            rr_irregularity = abs(prev_rr - next_rr) / max(rr_mean, 1e-6) if prev_rr > 0 and next_rr > 0 else 0.0
            rr_metadata.append([
                prev_rr,
                next_rr,
                rr_mean,
                rr_irregularity,
                float(beat_idx > 0),
                float(beat_idx + 1 < len(anchors)),
            ])
        metadata = torch.tensor(np.asarray(rr_metadata, dtype=np.float32), dtype=torch.float32)

        windows = []
        meta_kept = []
        anchors_kept = []
        window_starts = []
        window_ends = []
        for beat_idx, anchor in enumerate(anchors):
            start = anchor - pre
            end = start + window_len
            if start < 0 or end > prob_maps.shape[-1]:
                continue
            windows.append(prob_maps[idx:idx + 1, :, start:end])
            meta_kept.append(metadata[beat_idx])
            anchors_kept.append(anchor)
            window_starts.append(start)
            window_ends.append(end - 1)
        if not windows:
            peak = int(torch.argmax(prob_maps[idx, 1]).item())
            start = max(0, min(peak - pre, prob_maps.shape[-1] - window_len))
            end = start + window_len
            windows.append(prob_maps[idx:idx + 1, :, start:end])
            meta_kept.append(torch.zeros(6, dtype=torch.float32))
            anchors_kept.append(int(peak))
            window_starts.append(int(start))
            window_ends.append(int(end - 1))

        beat_prob_maps = torch.cat(windows, dim=0)
        beat_metadata = torch.stack(meta_kept, dim=0)
        beat_features = _classifier_features(
            beat_prob_maps,
            'seg_tokens_ctx_cal',
            qrs_anchor_index=pre,
            beat_metadata=beat_metadata,
            teacher_temperature=teacher_temperature,
        ).cpu()

        if feature_dim is None:
            feature_dim = beat_features.shape[1]
        max_beats = max(max_beats, beat_features.shape[0])
        all_instances.append(beat_features)
        all_labels.append(row['label'])
        stats_rows.append({
            'ecg_id': row['ecg_id'],
            'label': row['label'],
            'lead': lead,
            'ptbxl_record_path': row.get('record_path', ''),
            'n_predicted_beats': beat_features.shape[0],
            'mean_qrs_entropy': float(_normalized_entropy(prob_maps[idx:idx + 1]).mean().item()),
            'beat_anchor_samples': _serialize_positions(anchors_kept),
            'beat_window_starts': _serialize_positions(window_starts),
            'beat_window_ends': _serialize_positions(window_ends),
        })
        plot_rows.append({
            'ecg_id': row['ecg_id'],
            'label': row['label'],
            'record_path': row.get('record_path', ''),
            'anchor_samples': anchors_kept,
            'window_starts': window_starts,
            'window_ends': window_ends,
        })

    pooled_features = []
    padded_instances = []
    masks = []
    for beat_features in all_instances:
        pooled_features.append(torch.cat([beat_features.mean(dim=0), beat_features.max(dim=0).values], dim=0))
        padded = torch.zeros(max_beats, feature_dim, dtype=torch.float32)
        mask = torch.zeros(max_beats, dtype=torch.bool)
        padded[:beat_features.shape[0]] = beat_features
        mask[:beat_features.shape[0]] = True
        padded_instances.append(padded)
        masks.append(mask)

    return (
        torch.stack(pooled_features, dim=0),
        torch.stack(padded_instances, dim=0),
        torch.stack(masks, dim=0),
        torch.tensor(all_labels, dtype=torch.long),
        stats_rows,
        plot_rows,
    )


def _make_dataset(mode, features, instances, masks, labels):
    if mode == 'record_mil_ctx_cal':
        return TensorDataset(instances, masks, labels)
    return TensorDataset(features, labels)


def _run_epoch(model, loader, mode, optimizer=None):
    is_training = optimizer is not None
    model.train(is_training)
    total_loss = 0.0
    probs = []
    labels_all = []
    device = next(model.parameters()).device

    ctx = torch.enable_grad() if is_training else torch.no_grad()
    with ctx:
        for batch in loader:
            if mode == 'record_mil_ctx_cal':
                instances, mask, labels = batch
                instances = instances.to(device)
                mask = mask.to(device)
                labels = labels.to(device)
                logits = model(instances, mask)
                batch_size = len(instances)
            else:
                features, labels = batch
                features = features.to(device)
                labels = labels.to(device)
                logits = model(features)
                batch_size = len(features)
            loss = torch.nn.functional.cross_entropy(logits, labels)
            if is_training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * batch_size
            probs.append(torch.softmax(logits, dim=1)[:, 1].detach().cpu())
            labels_all.append(labels.detach().cpu())
    return {
        'loss': total_loss / len(loader.dataset),
        'probs': torch.cat(probs),
        'labels': torch.cat(labels_all),
    }


def _predict_with_metadata(model, loader, mode):
    model.eval()
    device = next(model.parameters()).device
    total_loss = 0.0
    probs = []
    labels_all = []
    attentions = []

    with torch.no_grad():
        for batch in loader:
            if mode == 'record_mil_ctx_cal':
                instances, mask, labels = batch
                instances = instances.to(device)
                mask = mask.to(device)
                labels = labels.to(device)
                logits, batch_attn = model(instances, mask, return_attention=True)
                for attn_row, mask_row in zip(batch_attn.cpu(), mask.cpu()):
                    attentions.append(attn_row[mask_row].numpy())
                batch_size = len(instances)
            else:
                features, labels = batch
                features = features.to(device)
                labels = labels.to(device)
                logits = model(features)
                batch_size = len(features)
            loss = torch.nn.functional.cross_entropy(logits, labels)
            total_loss += loss.item() * batch_size
            probs.append(torch.softmax(logits, dim=1)[:, 1].cpu())
            labels_all.append(labels.cpu())

    return {
        'loss': total_loss / len(loader.dataset),
        'probs': torch.cat(probs),
        'labels': torch.cat(labels_all),
        'attentions': attentions if attentions else None,
    }


def _build_cls_model(mode, train_inputs, device):
    if mode == 'record_mil_ctx_cal':
        return ECGRecordBagMILClassifier(instance_dim=train_inputs['instances'].shape[-1]).to(device)
    return ECGSegmenterFeatureClassifier(feature_dim=train_inputs['features'].shape[-1]).to(device)


def train_record_classifier(train_inputs, val_inputs, test_inputs, train_labels, val_labels, test_labels, args, seed, mode):
    set_seed(seed)
    device = get_device()
    model = _build_cls_model(mode, train_inputs, device)
    model.apply(initialize_he)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.cls_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    train_loader = DataLoader(_make_dataset(mode, train_inputs['features'], train_inputs['instances'], train_inputs['masks'], train_labels), batch_size=args.batch_size, shuffle=True)
    train_eval_loader = DataLoader(_make_dataset(mode, train_inputs['features'], train_inputs['instances'], train_inputs['masks'], train_labels), batch_size=args.batch_size, shuffle=False)
    val_loader = DataLoader(_make_dataset(mode, val_inputs['features'], val_inputs['instances'], val_inputs['masks'], val_labels), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(_make_dataset(mode, test_inputs['features'], test_inputs['instances'], test_inputs['masks'], test_labels), batch_size=args.batch_size, shuffle=False)

    rows = []
    best = None
    best_state_dict = None
    for epoch in range(1, args.epochs + 1):
        train_step = _run_epoch(model, train_loader, mode, optimizer=optimizer)
        train_eval = _run_epoch(model, train_eval_loader, mode)
        val_eval = _run_epoch(model, val_loader, mode)
        threshold, _ = _best_balanced_threshold(val_eval['probs'], val_eval['labels'])
        train_metrics = _binary_stats_extended(train_eval['probs'], train_eval['labels'], threshold)
        val_metrics = _binary_stats_extended(val_eval['probs'], val_eval['labels'], threshold)
        test_eval = _run_epoch(model, test_loader, mode)
        test_metrics = _binary_stats_extended(test_eval['probs'], test_eval['labels'], threshold)
        scheduler.step()

        row = {
            'epoch': epoch,
            'train_loss': train_step['loss'],
            'val_loss': val_eval['loss'],
            'test_loss': test_eval['loss'],
            'val_bal_acc': val_metrics['balanced_acc'],
            'test_bal_acc': test_metrics['balanced_acc'],
            'test_acc': test_metrics['accuracy'],
            'test_precision': test_metrics['precision'],
            'test_f1': test_metrics['f1'],
            'test_mcc': test_metrics['mcc'],
            'test_pr_auc': test_metrics['pr_auc'],
            'test_pos_recall': test_metrics['pos_recall'],
            'test_neg_recall': test_metrics['neg_recall'],
            'test_pred_pos_rate': test_metrics['pred_pos_rate'],
            'test_tp': test_metrics['tp'],
            'test_tn': test_metrics['tn'],
            'test_fp': test_metrics['fp'],
            'test_fn': test_metrics['fn'],
            'cls_threshold': threshold,
        }
        rows.append(row)
        if best is None or row['val_bal_acc'] > best['val_bal_acc']:
            best = dict(row)
            best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    checkpoint = {
        'mode': mode,
        'seed': seed,
        'teacher_temperature': best.get('teacher_temperature', None),
        'cls_threshold': best['cls_threshold'],
        'target_fs': args.target_fs,
        'source_fs': args.source_fs,
        'ptbxl_resolution': args.ptbxl_resolution,
        'lead': args.lead,
        'window_pre_ms': args.window_pre_ms,
        'window_post_ms': args.window_post_ms,
        'min_qrs_ms': args.min_qrs_ms,
        'feature_dim': int(train_inputs['features'].shape[-1]),
        'instance_dim': int(train_inputs['instances'].shape[-1]),
        'cls_model_state_dict': best_state_dict,
    }
    return rows, best, checkpoint


def evaluate_saved_classifier(inputs, labels, cls_checkpoint, mode):
    device = get_device()
    model = _build_cls_model(mode, inputs, device)
    model.load_state_dict(cls_checkpoint['cls_model_state_dict'])
    loader = DataLoader(_make_dataset(mode, inputs['features'], inputs['instances'], inputs['masks'], labels), batch_size=64, shuffle=False)
    result = _predict_with_metadata(model, loader, mode)
    metrics = _binary_stats_extended(result['probs'], result['labels'], cls_checkpoint['cls_threshold'])
    return {
        'loss': result['loss'],
        'metrics': metrics,
        'probs': result['probs'],
        'labels': result['labels'],
        'attentions': result['attentions'],
    }


def write_csv(path, rows):
    fieldnames = list(rows[0].keys())
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_record_visualizations(output_dir, split_name, split_payload, eval_output, cls_threshold, sample_rate, plot_limit):
    if plot_limit == 0:
        return
    plot_dir = os.path.join(output_dir, f'{split_name}_record_plots')
    os.makedirs(plot_dir, exist_ok=True)
    waves = split_payload['waves'][:, 0, :].cpu().numpy()
    prob_maps = split_payload['prob_maps']
    rows = split_payload['plot_rows']
    total_plots = len(rows) if plot_limit < 0 else min(plot_limit, len(rows))
    pred_probs = eval_output['probs'].cpu().numpy()
    pred_labels = (pred_probs >= cls_threshold).astype(np.int64)
    attentions = eval_output.get('attentions')

    for index in range(total_plots):
        plot_row = rows[index]
        attention = None if attentions is None else attentions[index]
        plot_path = os.path.join(plot_dir, f'{index:04d}_{_sample_name(plot_row)}.png')
        _plot_record_visualization(
            waves[index],
            prob_maps[index],
            plot_row['anchor_samples'],
            plot_row['window_starts'],
            plot_row['window_ends'],
            sample_rate,
            plot_row,
            plot_path,
            float(pred_probs[index]),
            int(pred_labels[index]),
            attention=attention,
        )
    if total_plots:
        print(f'Saved {split_name} record plots to: {plot_dir}')


def _augment_record_stats(stats, eval_output, cls_threshold):
    pred_probs = eval_output['probs'].cpu().numpy()
    pred_labels = (pred_probs >= cls_threshold).astype(np.int64)
    for row, pred_prob, pred_label in zip(stats, pred_probs, pred_labels):
        row['pred_prob'] = float(pred_prob)
        row['pred_label'] = int(pred_label)
        row['pred_label_name'] = _format_binary_label(pred_label)
        row['true_label_name'] = _format_binary_label(row['label'])


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seeds[0])

    rows = _load_ptbxl_rows(args.data_dir, args.pos_codes, args.neg_codes)
    rows = _filter_rows_by_available_resolution(rows, args.data_dir, args.restrict_to_available_resolution)
    train_rows, val_rows, test_rows = _split_ptbxl(rows)

    device = get_device()
    seg_payload = torch.load(args.seg_checkpoint, map_location='cpu')
    seg_model = ECGUNet3p(n_channels=seg_payload['args'].get('n_channels', 32)).to(device)
    seg_model.load_state_dict(seg_payload['seg_model_state_dict'])
    teacher_temperature = float(seg_payload.get('summary', {}).get('teacher_temperature', 1.0))

    def build_split(split_rows, name):
        X, probs, kept_rows = _segment_records(
            seg_model,
            split_rows,
            args.data_dir,
            args.lead,
            args.target_fs,
            args.source_fs,
            args.feature_batch_size,
            device,
            args.ptbxl_resolution,
        )
        features, instances, masks, labels, stats, plot_rows = _record_instances(
            probs,
            target_fs=args.target_fs,
            teacher_temperature=teacher_temperature,
            lead=args.lead,
            rows=kept_rows,
            window_pre_ms=args.window_pre_ms,
            window_post_ms=args.window_post_ms,
            min_qrs_ms=args.min_qrs_ms,
        )
        write_csv(os.path.join(args.output_dir, f'{name}_records.csv'), stats)
        return {
            'features': features,
            'instances': instances,
            'masks': masks,
            'waves': X,
            'prob_maps': probs,
            'plot_rows': plot_rows,
            'stats': stats,
        }, labels, len(kept_rows), len(split_rows)

    train_inputs, train_labels, train_avail, train_total = build_split(train_rows, 'train')
    val_inputs, val_labels, val_avail, val_total = build_split(val_rows, 'val')
    test_inputs, test_labels, test_avail, test_total = build_split(test_rows, 'test')
    print(
        f'PTB-XL availability: train {train_avail}/{train_total}, '
        f'val {val_avail}/{val_total}, test {test_avail}/{test_total}'
    )

    summary_rows = []
    best_export = None
    if args.cls_checkpoint is not None:
        print('Running final seg+cls inference with a frozen segmenter and saved downstream classifier.')
        cls_checkpoint = torch.load(args.cls_checkpoint, map_location='cpu')
        mode = cls_checkpoint['mode']
        val_eval = evaluate_saved_classifier(val_inputs, val_labels, cls_checkpoint, mode)
        test_eval = evaluate_saved_classifier(test_inputs, test_labels, cls_checkpoint, mode)
        _augment_record_stats(val_inputs['stats'], val_eval, cls_checkpoint['cls_threshold'])
        _augment_record_stats(test_inputs['stats'], test_eval, cls_checkpoint['cls_threshold'])
        write_csv(os.path.join(args.output_dir, 'val_records.csv'), val_inputs['stats'])
        write_csv(os.path.join(args.output_dir, 'test_records.csv'), test_inputs['stats'])
        _write_record_visualizations(
            args.output_dir, 'val', val_inputs, val_eval, cls_checkpoint['cls_threshold'], args.target_fs, args.plot_limit,
        )
        _write_record_visualizations(
            args.output_dir, 'test', test_inputs, test_eval, cls_checkpoint['cls_threshold'], args.target_fs, args.plot_limit,
        )
        row = {
            'epoch': -1,
            'train_loss': None,
            'val_loss': val_eval['loss'],
            'test_loss': test_eval['loss'],
            'val_bal_acc': val_eval['metrics']['balanced_acc'],
            'test_bal_acc': test_eval['metrics']['balanced_acc'],
            'test_acc': test_eval['metrics']['accuracy'],
            'test_precision': test_eval['metrics']['precision'],
            'test_f1': test_eval['metrics']['f1'],
            'test_mcc': test_eval['metrics']['mcc'],
            'test_pr_auc': test_eval['metrics']['pr_auc'],
            'test_pos_recall': test_eval['metrics']['pos_recall'],
            'test_neg_recall': test_eval['metrics']['neg_recall'],
            'test_pred_pos_rate': test_eval['metrics']['pred_pos_rate'],
            'test_tp': test_eval['metrics']['tp'],
            'test_tn': test_eval['metrics']['tn'],
            'test_fp': test_eval['metrics']['fp'],
            'test_fn': test_eval['metrics']['fn'],
            'cls_threshold': cls_checkpoint['cls_threshold'],
            'mode': mode,
            'seed': cls_checkpoint.get('seed', -1),
            'teacher_temperature': cls_checkpoint.get('teacher_temperature', teacher_temperature),
            'ptbxl_resolution': cls_checkpoint.get('ptbxl_resolution', args.ptbxl_resolution),
            'target_fs': cls_checkpoint.get('target_fs', args.target_fs),
            'source_fs': cls_checkpoint.get('source_fs', args.source_fs),
        }
        summary_rows.append(row)
        print(
            f"{mode} saved-cls: val_bal={row['val_bal_acc']:.4f} test_bal={row['test_bal_acc']:.4f} "
            f"precision={row['test_precision']:.4f} f1={row['test_f1']:.4f} "
            f"pr_auc={row['test_pr_auc']:.4f} mcc={row['test_mcc']:.4f}"
        )
    else:
        print('Training a new downstream classifier on top of frozen segmenter evidence.')
        for mode in args.modes:
            for seed in args.seeds:
                rows, best, checkpoint = train_record_classifier(
                    train_inputs, val_inputs, test_inputs, train_labels, val_labels, test_labels, args, seed, mode,
                )
                mode_seed_dir = os.path.join(args.output_dir, f'{mode}-seed{seed}')
                os.makedirs(mode_seed_dir, exist_ok=True)
                write_csv(os.path.join(mode_seed_dir, 'metrics.csv'), rows)
                torch.save(checkpoint, os.path.join(mode_seed_dir, 'best_cls_model.pt'))
                best['mode'] = mode
                best['seed'] = seed
                best['teacher_temperature'] = teacher_temperature
                best['ptbxl_resolution'] = args.ptbxl_resolution
                best['target_fs'] = args.target_fs
                best['source_fs'] = args.source_fs
                summary_rows.append(best)
                if best_export is None or best['val_bal_acc'] > best_export['row']['val_bal_acc']:
                    best_export = {'row': dict(best), 'checkpoint': checkpoint}
                print(
                    f"{mode} seed={seed}: val_bal={best['val_bal_acc']:.4f} test_bal={best['test_bal_acc']:.4f} "
                    f"precision={best['test_precision']:.4f} f1={best['test_f1']:.4f} "
                    f"pr_auc={best['test_pr_auc']:.4f} mcc={best['test_mcc']:.4f}"
                )
        if args.save_best_cls and best_export is not None:
            torch.save(best_export['checkpoint'], os.path.join(args.output_dir, 'best_cls_model.pt'))

    write_csv(os.path.join(args.output_dir, 'summary.csv'), summary_rows)
    print(f'Wrote summary to {os.path.join(args.output_dir, "summary.csv")}')


if __name__ == '__main__':
    main()
