import argparse
import csv
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from datareader import DEFAULT_LEADS, CLASS_NAMES, list_ludb_records, load_ludb_single_lead_tensors
from model import ECGUNet3p
from train import get_device, split_records


WAVE_CLASSES = [0, 1, 2]


def get_args():
    parser = argparse.ArgumentParser(description='Evaluate a single-lead ECG segmentation checkpoint on LUDB.')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to LUDB WFDB data directory')
    parser.add_argument('--seg-checkpoint', type=str, required=True, help='Path to segmenter checkpoint')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory for evaluation outputs')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--n_ludb_train', type=int, default=100)
    parser.add_argument('--eval-split', choices=['test', 'all'], default='test', help='Evaluate held-out records or all records')
    parser.add_argument('--n_channels', type=int, default=None)
    parser.add_argument('--target-fs', type=float, default=None, help='Optional target sample rate for resampling')
    parser.add_argument('--source-fs', type=float, default=None, help='Fallback source sample rate if the input records do not expose record.fs')
    parser.add_argument('--min-duration-ms', type=float, default=40.0)
    parser.add_argument('--boundary-tolerance-points', type=int, default=75, help='Boundary tolerance in datapoints; 75 = 150ms at 500Hz (AAMI standard)')
    parser.add_argument('--disable-post-processing', action='store_true', help='Disable paper-style post-processing of predicted labels')
    parser.add_argument('--plot-limit', type=int, default=-1, help='Number of interval comparison plots to save; -1 saves all, 0 disables plots')
    return parser.parse_args()


def segments_from_labels(labels, class_id, min_length=1):
    mask = labels == class_id
    segments = []
    start = None
    for index, value in enumerate(mask):
        if value and start is None:
            start = index
        elif not value and start is not None:
            if index - start >= min_length:
                segments.append((start, index - 1))
            start = None
    if start is not None and len(mask) - start >= min_length:
        segments.append((start, len(mask) - 1))
    return segments


def runs_from_labels(labels):
    runs = []
    start = 0
    current_label = int(labels[0])
    for index in range(1, len(labels)):
        label = int(labels[index])
        if label != current_label:
            runs.append((current_label, start, index - 1))
            start = index
            current_label = label
    runs.append((current_label, start, len(labels) - 1))
    return runs


def remove_short_runs(labels, min_length):
    cleaned = labels.copy()
    changed = True

    while changed:
        changed = False
        runs = runs_from_labels(cleaned)
        for run_index, (label, start, end) in enumerate(runs):
            length = end - start + 1
            if label == 3 or length >= min_length:
                continue

            previous_run = runs[run_index - 1] if run_index > 0 else None
            next_run = runs[run_index + 1] if run_index + 1 < len(runs) else None

            if previous_run is not None and next_run is not None:
                previous_label = previous_run[0]
                next_label = next_run[0]
                if previous_label == next_label:
                    replacement_label = previous_label
                else:
                    replacement_label = 3
            else:
                replacement_label = 3

            cleaned[start:end + 1] = replacement_label
            changed = True
            break

    return cleaned


def longest_segment_in_window(labels, class_id, window_start, window_end, min_length):
    if window_end < window_start:
        return None

    candidates = []
    for start, end in segments_from_labels(labels[window_start:window_end + 1], class_id, min_length=min_length):
        candidates.append((window_start + start, window_start + end))
    if not candidates:
        return None
    return max(candidates, key=lambda segment: segment[1] - segment[0] + 1)


def post_process_labels(labels, sample_rate, min_duration_ms):
    min_length = max(1, int(round(sample_rate * min_duration_ms / 1000.0)))
    cleaned = remove_short_runs(labels.astype(np.int64), min_length)
    processed = np.full_like(cleaned, fill_value=3)

    qrs_segments = segments_from_labels(cleaned, class_id=1, min_length=min_length)
    for start, end in qrs_segments:
        processed[start:end + 1] = 1

    if not qrs_segments:
        for class_id in [0, 2]:
            segment = longest_segment_in_window(cleaned, class_id, 0, len(cleaned) - 1, min_length)
            if segment is not None:
                processed[segment[0]:segment[1] + 1] = class_id
        return processed

    first_qrs_start = qrs_segments[0][0]
    p_before_first = longest_segment_in_window(cleaned, 0, 0, first_qrs_start - 1, min_length)
    if p_before_first is not None:
        processed[p_before_first[0]:p_before_first[1] + 1] = 0

    for left_qrs, right_qrs in zip(qrs_segments, qrs_segments[1:]):
        gap_start = left_qrs[1] + 1
        gap_end = right_qrs[0] - 1
        t_segment = longest_segment_in_window(cleaned, 2, gap_start, gap_end, min_length)
        p_segment = longest_segment_in_window(cleaned, 0, gap_start, gap_end, min_length)
        if t_segment is not None:
            processed[t_segment[0]:t_segment[1] + 1] = 2
        if p_segment is not None:
            processed[p_segment[0]:p_segment[1] + 1] = 0

    last_qrs_end = qrs_segments[-1][1]
    t_after_last = longest_segment_in_window(cleaned, 2, last_qrs_end + 1, len(cleaned) - 1, min_length)
    if t_after_last is not None:
        processed[t_after_last[0]:t_after_last[1] + 1] = 2

    return processed


def overlap_length(first, second):
    return max(0, min(first[1], second[1]) - max(first[0], second[0]) + 1)


def match_segments(true_segments, pred_segments):
    matched_pred = set()
    matches = []
    for true_segment in true_segments:
        best_index = None
        best_overlap = 0
        for pred_index, pred_segment in enumerate(pred_segments):
            if pred_index in matched_pred:
                continue
            overlap = overlap_length(true_segment, pred_segment)
            if overlap > best_overlap:
                best_overlap = overlap
                best_index = pred_index
        if best_index is not None and best_overlap > 0:
            matched_pred.add(best_index)
            matches.append((true_segment, pred_segments[best_index]))
    return matches


def summarize_errors(errors):
    if not errors:
        return {'mean_ms': None, 'std_ms': None, 'mae_ms': None}
    values = np.asarray(errors, dtype=np.float64)
    return {
        'mean_ms': float(values.mean()),
        'std_ms': float(values.std(ddof=0)),
        'mae_ms': float(np.abs(values).mean()),
    }


def compute_metrics(y_true, y_pred, sample_rate, min_duration_ms, boundary_tolerance_points):
    metrics = {}
    for class_id, class_name in enumerate(CLASS_NAMES):
        true_mask = y_true == class_id
        pred_mask = y_pred == class_id
        tp = int(np.logical_and(true_mask, pred_mask).sum())
        fp = int(np.logical_and(~true_mask, pred_mask).sum())
        fn = int(np.logical_and(true_mask, ~pred_mask).sum())
        metrics[class_name] = {
            'sample_precision': tp / (tp + fp) if (tp + fp) else None,
            'sample_recall': tp / (tp + fn) if (tp + fn) else None,
            'sample_dice': (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) else None,
            'sample_iou': tp / (tp + fp + fn) if (tp + fp + fn) else None,
        }

    min_length = max(1, int(round(sample_rate * min_duration_ms / 1000.0)))
    tolerance_ms = boundary_tolerance_points * 1000.0 / sample_rate

    for class_id in WAVE_CLASSES:
        class_name = CLASS_NAMES[class_id]
        true_count = 0
        pred_count = 0
        matched_count = 0
        onset_errors = []
        offset_errors = []
        onset_within_tolerance = 0
        offset_within_tolerance = 0
        both_within_tolerance = 0

        for true_record, pred_record in zip(y_true, y_pred):
            true_segments = segments_from_labels(true_record, class_id, min_length=1)
            pred_segments = segments_from_labels(pred_record, class_id, min_length=min_length)
            matches = match_segments(true_segments, pred_segments)
            true_count += len(true_segments)
            pred_count += len(pred_segments)
            matched_count += len(matches)

            for true_segment, pred_segment in matches:
                onset_error_points = pred_segment[0] - true_segment[0]
                offset_error_points = pred_segment[1] - true_segment[1]
                onset_errors.append(onset_error_points * 1000.0 / sample_rate)
                offset_errors.append(offset_error_points * 1000.0 / sample_rate)
                onset_hit = abs(onset_error_points) <= boundary_tolerance_points
                offset_hit = abs(offset_error_points) <= boundary_tolerance_points
                onset_within_tolerance += int(onset_hit)
                offset_within_tolerance += int(offset_hit)
                both_within_tolerance += int(onset_hit and offset_hit)

        # AAMI Table 4-style per-boundary Se / PPV / F1:
        #   Se  = within_tolerance / true_count   (how many true segments have boundary found)
        #   PPV = within_tolerance / pred_count   (how many predicted boundaries are correct)
        #   F1  = 2 * within_tolerance / (true_count + pred_count)
        onset_se  = onset_within_tolerance / true_count if true_count else None
        onset_ppv = onset_within_tolerance / pred_count if pred_count else None
        onset_f1  = 2 * onset_within_tolerance / (true_count + pred_count) if (true_count + pred_count) else None
        offset_se  = offset_within_tolerance / true_count if true_count else None
        offset_ppv = offset_within_tolerance / pred_count if pred_count else None
        offset_f1  = 2 * offset_within_tolerance / (true_count + pred_count) if (true_count + pred_count) else None

        metrics[class_name].update({
            'segment_sensitivity': matched_count / true_count if true_count else None,
            'segment_ppv': matched_count / pred_count if pred_count else None,
            'segment_f1': (2 * matched_count / (true_count + pred_count)) if (true_count + pred_count) else None,
            'true_segments': true_count,
            'pred_segments': pred_count,
            'matched_segments': matched_count,
            'onset_error': summarize_errors(onset_errors),
            'offset_error': summarize_errors(offset_errors),
            'boundary_tolerance_points': boundary_tolerance_points,
            'boundary_tolerance_ms': tolerance_ms,
            'onset_within_tolerance_rate': onset_within_tolerance / true_count if true_count else None,
            'offset_within_tolerance_rate': offset_within_tolerance / true_count if true_count else None,
            'both_boundaries_within_tolerance_rate': both_within_tolerance / true_count if true_count else None,
            'onset_within_tolerance_matched_rate': onset_within_tolerance / matched_count if matched_count else None,
            'offset_within_tolerance_matched_rate': offset_within_tolerance / matched_count if matched_count else None,
            'both_boundaries_within_tolerance_matched_rate': both_within_tolerance / matched_count if matched_count else None,
            'onset_se': onset_se,
            'onset_ppv': onset_ppv,
            'onset_f1': onset_f1,
            'offset_se': offset_se,
            'offset_ppv': offset_ppv,
            'offset_f1': offset_f1,
        })

    metrics['overall'] = {'sample_accuracy': float((y_true == y_pred).mean())}
    return metrics


def write_beat_intervals_csv(all_beats, sample_names, output_path):
    """Write one row per beat across all samples."""
    fieldnames = [
        'sample', 'beat',
        'qrs_on_ms', 'qrs_off_ms', 'qrs_duration_ms', 'r_peak_value',
        'p_on_ms', 'p_off_ms', 'p_duration_ms', 'p_peak_value', 'pr_interval_ms',
        't_on_ms', 't_off_ms', 't_duration_ms', 't_peak_value', 'qt_interval_ms',
    ]
    with open(output_path, 'w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for sample_name, beats in zip(sample_names, all_beats):
            for beat in beats:
                row = {'sample': sample_name}
                row.update(beat)
                writer.writerow(row)


def write_boundary_metrics(metrics, output_path):
    """Write AAMI Table 4-style per-boundary metrics: one row per (wave, boundary)."""
    fieldnames = [
        'wave', 'boundary',
        'se', 'ppv', 'f1',
        'mean_ms', 'std_ms', 'mae_ms',
        'within_tolerance_rate', 'boundary_tolerance_ms', 'boundary_tolerance_points',
        'true_segments', 'pred_segments',
    ]
    with open(output_path, 'w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for wave in CLASS_NAMES[:-1]:
            m = metrics[wave]
            for boundary in ('onset', 'offset'):
                writer.writerow({
                    'wave': wave,
                    'boundary': boundary,
                    'se':  m[f'{boundary}_se'],
                    'ppv': m[f'{boundary}_ppv'],
                    'f1':  m[f'{boundary}_f1'],
                    'mean_ms': m[f'{boundary}_error']['mean_ms'],
                    'std_ms':  m[f'{boundary}_error']['std_ms'],
                    'mae_ms':  m[f'{boundary}_error']['mae_ms'],
                    'within_tolerance_rate': m[f'{boundary}_within_tolerance_rate'],
                    'boundary_tolerance_ms': m['boundary_tolerance_ms'],
                    'boundary_tolerance_points': m['boundary_tolerance_points'],
                    'true_segments': m['true_segments'],
                    'pred_segments': m['pred_segments'],
                })


def write_wave_metrics(metrics, output_path):
    fieldnames = [
        'wave', 'sample_dice', 'sample_iou', 'segment_sensitivity', 'segment_ppv', 'segment_f1',
        'onset_mean_ms', 'onset_std_ms', 'onset_mae_ms',
        'offset_mean_ms', 'offset_std_ms', 'offset_mae_ms',
        'onset_within_tolerance_rate', 'offset_within_tolerance_rate', 'both_boundaries_within_tolerance_rate',
        'onset_within_tolerance_matched_rate', 'offset_within_tolerance_matched_rate',
        'both_boundaries_within_tolerance_matched_rate', 'boundary_tolerance_points', 'boundary_tolerance_ms',
        'true_segments', 'pred_segments', 'matched_segments',
    ]
    with open(output_path, 'w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for wave in CLASS_NAMES[:-1]:
            wave_metrics = metrics[wave]
            writer.writerow({
                'wave': wave,
                'sample_dice': wave_metrics['sample_dice'],
                'sample_iou': wave_metrics['sample_iou'],
                'segment_sensitivity': wave_metrics['segment_sensitivity'],
                'segment_ppv': wave_metrics['segment_ppv'],
                'segment_f1': wave_metrics['segment_f1'],
                'onset_mean_ms': wave_metrics['onset_error']['mean_ms'],
                'onset_std_ms': wave_metrics['onset_error']['std_ms'],
                'onset_mae_ms': wave_metrics['onset_error']['mae_ms'],
                'offset_mean_ms': wave_metrics['offset_error']['mean_ms'],
                'offset_std_ms': wave_metrics['offset_error']['std_ms'],
                'offset_mae_ms': wave_metrics['offset_error']['mae_ms'],
                'onset_within_tolerance_rate': wave_metrics['onset_within_tolerance_rate'],
                'offset_within_tolerance_rate': wave_metrics['offset_within_tolerance_rate'],
                'both_boundaries_within_tolerance_rate': wave_metrics['both_boundaries_within_tolerance_rate'],
                'onset_within_tolerance_matched_rate': wave_metrics['onset_within_tolerance_matched_rate'],
                'offset_within_tolerance_matched_rate': wave_metrics['offset_within_tolerance_matched_rate'],
                'both_boundaries_within_tolerance_matched_rate': wave_metrics['both_boundaries_within_tolerance_matched_rate'],
                'boundary_tolerance_points': wave_metrics['boundary_tolerance_points'],
                'boundary_tolerance_ms': wave_metrics['boundary_tolerance_ms'],
                'true_segments': wave_metrics['true_segments'],
                'pred_segments': wave_metrics['pred_segments'],
                'matched_segments': wave_metrics['matched_segments'],
            })


def build_sample_names(records, leads):
    sample_names = []
    for record in records:
        record_name = os.path.basename(record)
        for lead in leads:
            sample_names.append(f'{record_name}_{lead}')
    return sample_names


def records_from_checkpoint_split(records, checkpoint_args, split_name):
    record_names = checkpoint_args.get(f'{split_name}_records')
    if not record_names:
        return None

    record_by_name = {os.path.basename(record): record for record in records}
    return [record_by_name[name] for name in record_names]


def select_eval_records(records, checkpoint_args, eval_split, n_ludb_train):
    if eval_split == 'all':
        return records

    checkpoint_records = records_from_checkpoint_split(records, checkpoint_args, 'test')
    if checkpoint_records is not None:
        return checkpoint_records

    return records[n_ludb_train:]


def add_interval_spans(axis, labels, title, sample_rate, min_length=1):
    from matplotlib.patches import Patch

    colors = {
        0: 'red',
        1: 'green',
        2: 'purple',
    }
    for class_id, color in colors.items():
        for start, end in segments_from_labels(labels, class_id, min_length=min_length):
            axis.axvspan(start / sample_rate, end / sample_rate, color=color, alpha=0.25, linewidth=0)
    handles = [Patch(facecolor=colors[class_id], alpha=0.25, label=CLASS_NAMES[class_id]) for class_id in colors]
    axis.set_title(title)
    axis.legend(handles=handles, loc='upper right', ncol=3, fontsize='small')


def compute_beat_intervals(wave, labels, sample_rate, min_duration_ms):
    """Per-beat clinical intervals anchored on QRS complexes.

    Returns a list of dicts, one per detected QRS beat, with keys:
      beat, qrs_on_ms, qrs_off_ms, qrs_duration_ms, r_peak_value,
      p_on_ms, p_off_ms, p_duration_ms, p_peak_value, pr_interval_ms,
      t_on_ms, t_off_ms, t_duration_ms, t_peak_value, qt_interval_ms.
    """
    min_length = max(1, int(round(sample_rate * min_duration_ms / 1000.0)))
    qrs_segments = segments_from_labels(labels, class_id=1, min_length=min_length)
    beats = []
    for beat_idx, (qrs_on, qrs_off) in enumerate(qrs_segments):
        beat = {
            'beat': beat_idx + 1,
            'qrs_on': qrs_on,
            'qrs_off': qrs_off,
            'qrs_on_ms': qrs_on * 1000.0 / sample_rate,
            'qrs_off_ms': qrs_off * 1000.0 / sample_rate,
            'qrs_duration_ms': (qrs_off - qrs_on + 1) * 1000.0 / sample_rate,
            'r_peak_value': float(wave[qrs_on:qrs_off + 1].max()),
        }

        # P wave: search up to 400 ms before QRS onset
        p_search_start = max(0, qrs_on - int(round(sample_rate * 0.4)))
        p_segs = segments_from_labels(labels[p_search_start:qrs_on], class_id=0, min_length=1)
        if p_segs:
            p_rel_on, p_rel_off = p_segs[-1]  # closest P to QRS
            p_on = p_search_start + p_rel_on
            p_off = p_search_start + p_rel_off
            beat.update({
                'p_on': p_on,
                'p_off': p_off,
                'p_on_ms': p_on * 1000.0 / sample_rate,
                'p_off_ms': p_off * 1000.0 / sample_rate,
                'p_duration_ms': (p_off - p_on + 1) * 1000.0 / sample_rate,
                'pr_interval_ms': (qrs_on - p_on) * 1000.0 / sample_rate,
                'p_peak_value': float(wave[p_on:p_off + 1].max()),
            })
        else:
            beat.update({'p_on': None, 'p_off': None, 'p_on_ms': None, 'p_off_ms': None,
                         'p_duration_ms': None, 'pr_interval_ms': None, 'p_peak_value': None})

        # T wave: search up to 600 ms after QRS offset
        t_search_end = min(len(labels), qrs_off + 1 + int(round(sample_rate * 0.6)))
        t_segs = segments_from_labels(labels[qrs_off + 1:t_search_end], class_id=2, min_length=1)
        if t_segs:
            t_rel_on, t_rel_off = t_segs[0]  # first T after QRS
            t_on = qrs_off + 1 + t_rel_on
            t_off = qrs_off + 1 + t_rel_off
            beat.update({
                't_on': t_on,
                't_off': t_off,
                't_on_ms': t_on * 1000.0 / sample_rate,
                't_off_ms': t_off * 1000.0 / sample_rate,
                't_duration_ms': (t_off - t_on + 1) * 1000.0 / sample_rate,
                'qt_interval_ms': (t_off - qrs_on) * 1000.0 / sample_rate,
                't_peak_value': float(wave[t_on:t_off + 1].max()),
            })
        else:
            beat.update({'t_on': None, 't_off': None, 't_on_ms': None, 't_off_ms': None,
                         't_duration_ms': None, 'qt_interval_ms': None, 't_peak_value': None})

        beats.append(beat)
    return beats


def plot_beat_analysis(
    wave,
    labels,
    probs,
    beat_intervals,
    sample_rate,
    sample_name,
    output_path,
):
    """Two-panel beat analysis plot.

    Top panel: ECG signal with P/QRS/T colored spans and per-beat clinical
    interval annotations (QRS duration, PR interval, QT interval).
    Bottom panel: per-class softmax probability curves.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    time_axis = np.arange(len(wave)) / sample_rate
    colors = {0: 'red', 1: 'green', 2: 'purple'}
    class_labels = {0: 'P', 1: 'QRS', 2: 'T'}

    has_probs = probs is not None
    n_rows = 2 if has_probs else 1
    height_ratios = [3, 1] if has_probs else [1]
    fig, axes = plt.subplots(
        n_rows, 1,
        figsize=(18, 7 if has_probs else 4),
        sharex=True,
        gridspec_kw={'height_ratios': height_ratios},
        constrained_layout=True,
    )
    if n_rows == 1:
        axes = [axes]

    # ── top panel: ECG + intervals + beat annotations ──
    ax = axes[0]
    ax.plot(time_axis, wave, color='black', linewidth=0.8, zorder=2)
    for class_id, color in colors.items():
        for start, end in segments_from_labels(labels, class_id, min_length=1):
            ax.axvspan(start / sample_rate, end / sample_rate,
                       color=color, alpha=0.20, linewidth=0, zorder=1)
    handles = [Patch(facecolor=colors[c], alpha=0.4, label=class_labels[c]) for c in colors]
    ax.legend(handles=handles, loc='upper right', ncol=3, fontsize='small')
    ax.set_ylabel('Amplitude')
    ax.grid(alpha=0.2)

    # annotate each beat
    y_top = ax.get_ylim()[1]
    for beat in beat_intervals:
        qrs_mid = (beat['qrs_on'] + beat['qrs_off']) / 2 / sample_rate
        lines = [f"B{beat['beat']}  QRS:{beat['qrs_duration_ms']:.0f}ms"]
        if beat.get('pr_interval_ms') is not None:
            lines.append(f"PR:{beat['pr_interval_ms']:.0f}ms")
        if beat.get('qt_interval_ms') is not None:
            lines.append(f"QT:{beat['qt_interval_ms']:.0f}ms")
        ax.text(
            qrs_mid, y_top, '\n'.join(lines),
            ha='center', va='top', fontsize=6,
            color='#333333',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.6, linewidth=0),
        )
    ax.set_title(f'{sample_name} — per-beat clinical intervals')

    # ── bottom panel: probability curves ──
    if has_probs:
        ax2 = axes[1]
        prob_colors = {0: 'red', 1: 'green', 2: 'purple'}
        prob_labels = {0: 'P prob', 1: 'QRS prob', 2: 'T prob'}
        for class_id in [0, 1, 2]:
            ax2.plot(time_axis, probs[class_id], color=prob_colors[class_id],
                     linewidth=0.8, label=prob_labels[class_id], alpha=0.8)
        ax2.set_ylim(-0.05, 1.05)
        ax2.set_ylabel('Probability')
        ax2.legend(loc='upper right', ncol=3, fontsize='small')
        ax2.grid(alpha=0.2)
        ax2.set_xlabel('Time (s)')
    else:
        axes[0].set_xlabel('Time (s)')

    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_interval_comparisons(
    waves,
    y_true,
    y_raw_pred,
    y_current_pred,
    sample_rate,
    output_dir,
    sample_names,
    plot_limit,
):
    if plot_limit == 0:
        return None

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plot_dir = os.path.join(output_dir, 'interval_plots')
    os.makedirs(plot_dir, exist_ok=True)

    total_plots = len(waves) if plot_limit < 0 else min(plot_limit, len(waves))
    for index in range(total_plots):
        wave = waves[index]
        time_axis = np.arange(len(wave)) / sample_rate

        fig, axes = plt.subplots(3, 1, figsize=(18, 9), sharex=True, constrained_layout=True)
        rows = [
            ('Ground truth intervals', y_true[index], 1),
            ('Original raw prediction intervals', y_raw_pred[index], 1),
            ('Current evaluated intervals', y_current_pred[index], 1),
        ]

        for axis, (title, labels, row_min_length) in zip(axes, rows):
            axis.plot(time_axis, wave, color='black', linewidth=0.8)
            add_interval_spans(axis, labels, title, sample_rate, min_length=row_min_length)
            axis.set_ylabel('ECG')
            axis.grid(alpha=0.2)

        axes[-1].set_xlabel('Time (s)')
        fig.suptitle(f'{sample_names[index]} | P=red, QRS=green, T=purple', fontsize=14)
        fig.savefig(os.path.join(plot_dir, f'{index:04d}_{sample_names[index]}.png'), dpi=150)
        plt.close(fig)

    return plot_dir


def evaluate_checkpoint(args, checkpoint, checkpoint_args, records):
    test_records = select_eval_records(records, checkpoint_args, args.eval_split, args.n_ludb_train)
    if not test_records:
        print(
            f'Warning: n_ludb_train={args.n_ludb_train} leaves no held-out test records; '
            'evaluating all records instead. Use --eval-split all to make this explicit.'
        )
        test_records = records

    target_fs = args.target_fs or checkpoint_args.get('sample_rate') or 500
    n_channels = args.n_channels or checkpoint_args.get('n_channels', 32)
    X_test, y_test, sample_rate = load_ludb_single_lead_tensors(
        test_records, leads=DEFAULT_LEADS, target_fs=target_fs, source_fs=args.source_fs,
    )

    device = get_device()
    print(f'Using device: {device}')
    print(f'Using sample rate: {sample_rate:g} Hz')
    print(f'Segmentation samples: {tuple(X_test.shape)}')

    seg_model = ECGUNet3p(n_channels=n_channels).to(device)
    seg_model.load_state_dict(checkpoint['seg_model_state_dict'])

    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=args.batch_size, shuffle=False)
    raw_pred_labels = []
    pred_probs_list = []
    true_labels = []
    seg_model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            raw_output = seg_model(data)
            raw_pred_labels.append(raw_output.argmax(dim=1).cpu().numpy())
            pred_probs_list.append(F.softmax(raw_output, dim=1).cpu().numpy())
            true_labels.append(target.argmax(dim=1).numpy())

    y_raw_pred = np.concatenate(raw_pred_labels, axis=0)
    y_pred_probs = np.concatenate(pred_probs_list, axis=0)
    y_true = np.concatenate(true_labels, axis=0)

    post_processing_enabled = not args.disable_post_processing
    if post_processing_enabled:
        y_pred = np.stack([
            post_process_labels(record, sample_rate, args.min_duration_ms)
            for record in y_raw_pred
        ], axis=0)
    else:
        y_pred = y_raw_pred

    sample_names = build_sample_names(test_records, DEFAULT_LEADS)
    plot_dir = plot_interval_comparisons(
        X_test[:, 0, :].numpy(),
        y_true,
        y_raw_pred,
        y_pred,
        sample_rate,
        args.output_dir,
        sample_names,
        args.plot_limit,
    )
    if plot_dir is not None:
        print(f'Saved interval comparison plots to: {plot_dir}')

    metrics = compute_metrics(
        y_true,
        y_pred,
        sample_rate,
        args.min_duration_ms,
        args.boundary_tolerance_points,
    )
    metrics['checkpoint'] = {
        'seg_path': args.seg_checkpoint,
        'epoch': checkpoint.get('epoch'),
        'model_name': checkpoint_args.get('model_name', 'ECGUNet3p'),
    }
    metrics['post_processing'] = {
        'enabled': post_processing_enabled,
        'min_duration_ms': args.min_duration_ms,
        'method': 'remove short non-none runs, keep QRS anchors, keep longest P/T in QRS intervals',
    }
    metrics['cross_task_guidance'] = {
        'inference_modifies_segmentation': False,
    }

    write_evaluation_outputs(args, metrics, X_test, y_pred, y_pred_probs, sample_names, sample_rate)
    print(json.dumps(metrics, indent=2))


def write_evaluation_outputs(args, metrics, X_test, y_pred, y_pred_probs, sample_names, sample_rate):
    with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as metrics_file:
        json.dump(metrics, metrics_file, indent=2)
    write_wave_metrics(metrics, os.path.join(args.output_dir, 'wave_metrics.csv'))
    write_boundary_metrics(metrics, os.path.join(args.output_dir, 'boundary_metrics.csv'))

    waves_np = X_test[:, 0, :].numpy()
    all_beats = [
        compute_beat_intervals(wave, labels, sample_rate, args.min_duration_ms)
        for wave, labels in zip(waves_np, y_pred)
    ]
    write_beat_intervals_csv(
        all_beats,
        sample_names,
        os.path.join(args.output_dir, 'beat_intervals.csv'),
    )
    plot_beat_dir = os.path.join(args.output_dir, 'beat_plots')
    os.makedirs(plot_beat_dir, exist_ok=True)
    total_plots = len(waves_np) if args.plot_limit < 0 else min(args.plot_limit, len(waves_np))
    for index in range(total_plots):
        plot_beat_analysis(
            waves_np[index],
            y_pred[index],
            y_pred_probs[index] if y_pred_probs is not None else None,
            all_beats[index],
            sample_rate,
            sample_names[index],
            os.path.join(plot_beat_dir, f'{index:04d}_{sample_names[index]}.png'),
        )
    if total_plots:
        print(f'Saved beat analysis plots to: {plot_beat_dir}')


def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)

    checkpoint = torch.load(args.seg_checkpoint, map_location='cpu')
    checkpoint_args = checkpoint.get('args', {})
    records = list_ludb_records(args.data_dir)
    if 'seg_model_state_dict' not in checkpoint:
        raise ValueError('Segmenter checkpoint must contain seg_model_state_dict.')
    evaluate_checkpoint(args, checkpoint, checkpoint_args, records)


if __name__ == '__main__':
    main()
