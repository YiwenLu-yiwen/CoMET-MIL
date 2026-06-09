import os
from fractions import Fraction

import numpy as np
import torch
import wfdb
from scipy.signal import resample_poly


DEFAULT_LEADS = ('i', 'ii', 'iii', 'avr', 'avl', 'avf', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6')
CLASS_NAMES = ('p', 'qrs', 't', 'none')
AF_AFL_RHYTHMS = ('Atrial fibrillation', 'Atrial flutter, typical')


def normalize_waveform(wave):
    wave = np.asarray(wave, dtype=np.float32)
    centered_wave = wave - np.mean(wave)
    max_abs = np.max(np.abs(centered_wave))
    if max_abs == 0:
        return centered_wave
    return centered_wave / max_abs


def resample_waveform(wave, source_fs, target_fs):
    if target_fs is None or float(source_fs) == float(target_fs):
        return np.asarray(wave, dtype=np.float32)

    ratio = Fraction(float(target_fs) / float(source_fs)).limit_denominator(1000)
    resampled = resample_poly(wave, ratio.numerator, ratio.denominator)
    target_length = int(round(len(wave) * float(target_fs) / float(source_fs)))
    return resampled[:target_length].astype(np.float32)


def resolve_source_fs(record, fallback_source_fs=None, record_name=None):
    record_fs = getattr(record, 'fs', None)
    if record_fs is not None:
        return float(record_fs)
    if fallback_source_fs is not None:
        return float(fallback_source_fs)
    record_label = record_name or '<unknown record>'
    raise ValueError(
        f'Missing source sampling rate for {record_label}. '
        'Pass --source-fs explicitly when the source files do not expose record.fs.'
    )


def record_sort_key(record_path):
    basename = os.path.basename(record_path)
    try:
        return int(basename)
    except ValueError:
        return basename


def list_ludb_records(data_dir):
    return sorted(
        (
            os.path.abspath(os.path.join(data_dir, path[:-4]))
            for path in os.listdir(data_dir)
            if path.endswith('.hea')
        ),
        key=record_sort_key,
    )


def get_ludb_rhythm(record_name):
    record = wfdb.rdrecord(record_name)
    for comment in record.comments:
        if comment.startswith('Rhythm:'):
            return comment.replace('Rhythm:', '', 1).strip().rstrip('.')
    return None


def rhythm_to_af_afl_label(rhythm):
    return int(rhythm in AF_AFL_RHYTHMS)


def get_ludb_af_afl_label(record_name):
    return rhythm_to_af_afl_label(get_ludb_rhythm(record_name))


def annotation_intervals(record_name, lead):
    annotation = wfdb.rdann(record_name, extension=lead)
    intervals = {'p': [], 'qrs': [], 't': []}
    onset = None
    current_key = None

    for sample, symbol in zip(annotation.sample, annotation.symbol):
        if symbol == '(':
            onset = sample
        elif symbol == ')':
            if onset is not None and current_key is not None:
                intervals[current_key].append((onset, sample))
            onset = None
        elif symbol in {'p', 't'}:
            current_key = symbol
        else:
            assert symbol == 'N'
            current_key = 'qrs'

    return intervals


def intervals_to_target(intervals, source_fs, target_fs, target_length):
    labels = {name: np.zeros(target_length, dtype=np.float32) for name in CLASS_NAMES[:-1]}
    scale = float(target_fs) / float(source_fs)

    for class_name, class_intervals in intervals.items():
        for onset, offset in class_intervals:
            start = int(round(onset * scale))
            end = int(round(offset * scale))
            start = max(0, min(start, target_length - 1))
            end = max(start, min(end, target_length - 1))
            labels[class_name][start:end + 1] = 1.0

    occupied = labels['p'] + labels['qrs'] + labels['t']
    assert np.max(occupied) <= 1.0
    labels['none'] = 1.0 - occupied
    return np.stack([labels[name] for name in CLASS_NAMES], axis=0)


def load_ludb_single_lead_tensors(
    record_names,
    leads=DEFAULT_LEADS,
    target_fs=None,
    include_rhythm_label=False,
    source_fs=None,
):
    if not record_names:
        raise ValueError('No LUDB records were provided to load_ludb_single_lead_tensors.')

    waves = []
    targets = []
    rhythm_targets = []
    output_fs = None

    for record_name in record_names:
        record = wfdb.rdrecord(record_name)
        resolved_source_fs = resolve_source_fs(record, fallback_source_fs=source_fs, record_name=record_name)
        output_fs = float(target_fs or resolved_source_fs)
        lead_to_wave = dict(zip(record.sig_name, record.p_signal.T))
        rhythm_target = get_ludb_af_afl_label(record_name) if include_rhythm_label else None

        for lead in leads:
            wave = resample_waveform(lead_to_wave[lead], resolved_source_fs, output_fs)
            wave = normalize_waveform(wave)
            intervals = annotation_intervals(record_name, lead)
            target = intervals_to_target(intervals, resolved_source_fs, output_fs, len(wave))
            waves.append(wave)
            targets.append(target)
            if include_rhythm_label:
                rhythm_targets.append(rhythm_target)

    X = torch.tensor(np.asarray(waves), dtype=torch.float32).unsqueeze(1)
    y = torch.tensor(np.asarray(targets), dtype=torch.float32)
    if include_rhythm_label:
        y_rhythm = torch.tensor(np.asarray(rhythm_targets), dtype=torch.long)
        return X, y, y_rhythm, output_fs
    return X, y, output_fs


def load_ludb_window_tensors(
    record_names,
    leads=DEFAULT_LEADS,
    target_fs=None,
    source_fs=None,
    window_pre_ms=300,
    window_post_ms=80,
    p_min_overlap_ms=20,
    p_label_post_ms=40,
    return_seg_targets=False,
    return_metadata=False,
    return_keys=False,
):
    """Extract per-beat windows anchored on QRS onset.

    For each QRS interval in each lead, a fixed window of
    ``[qrs_onset - window_pre_ms, qrs_onset + window_post_ms]`` is extracted.
    The label is P-present (1) only if an annotated P interval overlaps the
    search region ``[qrs_onset - window_pre_ms, qrs_onset + p_label_post_ms]``
    by at least ``p_min_overlap_ms``. This avoids treating tiny edge contact as
    a positive label.

    Returns
    -------
    X_windows : torch.Tensor  shape [N, 1, window_samples]
        Normalised single-lead ECG windows.
    y_p_present : torch.Tensor  shape [N]  dtype long
        Per-window P-present (1) / P-absent (0) labels.
    window_samples : int
        Fixed window length in samples (at target_fs).
    output_fs : float
    """
    if not record_names:
        raise ValueError('No LUDB records provided to load_ludb_window_tensors.')

    windows_list = []
    labels_list = []
    seg_targets_list = []
    metadata_list = []
    key_list = []
    output_fs = None

    for record_name in record_names:
        record = wfdb.rdrecord(record_name)
        resolved_source_fs = resolve_source_fs(record, fallback_source_fs=source_fs, record_name=record_name)
        output_fs = float(target_fs or resolved_source_fs)
        lead_to_wave = dict(zip(record.sig_name, record.p_signal.T))

        pre_samples_src = int(round(window_pre_ms * resolved_source_fs / 1000.0))
        post_samples_src = int(round(window_post_ms * resolved_source_fs / 1000.0))
        label_post_samples_src = int(round(p_label_post_ms * resolved_source_fs / 1000.0))
        pre_samples_tgt = int(round(window_pre_ms * output_fs / 1000.0))
        post_samples_tgt = int(round(window_post_ms * output_fs / 1000.0))
        p_min_overlap_samples_tgt = max(1, int(round(p_min_overlap_ms * output_fs / 1000.0)))
        window_len = pre_samples_tgt + post_samples_tgt

        for lead in leads:
            raw_wave = lead_to_wave[lead]
            wave = resample_waveform(raw_wave, resolved_source_fs, output_fs)
            wave = normalize_waveform(wave)
            intervals = annotation_intervals(record_name, lead)
            scale = output_fs / resolved_source_fs
            qrs_onsets_tgt = [int(round(qrs_onset_src * scale)) for qrs_onset_src, _ in intervals['qrs']]
            dense_target = None
            if return_seg_targets:
                dense_target = intervals_to_target(intervals, resolved_source_fs, output_fs, len(wave))

            for beat_index, (qrs_onset_src, _) in enumerate(intervals['qrs']):
                win_start_src = qrs_onset_src - pre_samples_src

                win_start = int(round(win_start_src * scale))
                win_end = win_start + window_len

                if win_start < 0 or win_end > len(wave):
                    continue

                win = wave[win_start:win_end]
                if len(win) != window_len:
                    continue

                label_start = win_start
                label_end = int(round((qrs_onset_src + label_post_samples_src) * scale))
                label_end = max(label_start, min(label_end, win_end - 1))

                p_present = 0
                for p_onset_src, p_offset_src in intervals['p']:
                    p_onset_tgt = int(round(p_onset_src * scale))
                    p_offset_tgt = int(round(p_offset_src * scale))
                    overlap = max(0, min(p_offset_tgt, label_end) - max(p_onset_tgt, label_start) + 1)
                    if overlap >= p_min_overlap_samples_tgt:
                        p_present = 1
                        break

                windows_list.append(win)
                labels_list.append(p_present)
                if return_seg_targets:
                    seg_targets_list.append(dense_target[:, win_start:win_end])
                if return_metadata:
                    current_qrs_tgt = qrs_onsets_tgt[beat_index]
                    prev_rr = (
                        (current_qrs_tgt - qrs_onsets_tgt[beat_index - 1]) / output_fs
                        if beat_index > 0 else 0.0
                    )
                    next_rr = (
                        (qrs_onsets_tgt[beat_index + 1] - current_qrs_tgt) / output_fs
                        if beat_index + 1 < len(qrs_onsets_tgt) else 0.0
                    )
                    rr_values = [value for value in (prev_rr, next_rr) if value > 0]
                    rr_mean = float(np.mean(rr_values)) if rr_values else 0.0
                    rr_irregularity = (
                        abs(prev_rr - next_rr) / max(rr_mean, 1e-6)
                        if prev_rr > 0 and next_rr > 0 else 0.0
                    )
                    metadata_list.append([
                        prev_rr,
                        next_rr,
                        rr_mean,
                        rr_irregularity,
                        float(beat_index > 0),
                        float(beat_index + 1 < len(qrs_onsets_tgt)),
                    ])
                if return_keys:
                    key_list.append((os.path.basename(record_name), lead, beat_index))

    if not windows_list:
        raise ValueError('No valid windows extracted. Check record paths and lead names.')

    X_windows = torch.tensor(np.asarray(windows_list, dtype=np.float32), dtype=torch.float32).unsqueeze(1)
    y_p_present = torch.tensor(labels_list, dtype=torch.long)
    metadata = None
    if return_metadata:
        metadata = torch.tensor(np.asarray(metadata_list, dtype=np.float32), dtype=torch.float32)
    if return_seg_targets:
        y_seg_windows = torch.tensor(np.asarray(seg_targets_list, dtype=np.float32), dtype=torch.float32)
        if return_metadata:
            if return_keys:
                return X_windows, y_p_present, y_seg_windows, metadata, window_len, output_fs, key_list
            return X_windows, y_p_present, y_seg_windows, metadata, window_len, output_fs
        if return_keys:
            return X_windows, y_p_present, y_seg_windows, window_len, output_fs, key_list
        return X_windows, y_p_present, y_seg_windows, window_len, output_fs
    if return_metadata:
        if return_keys:
            return X_windows, y_p_present, metadata, window_len, output_fs, key_list
        return X_windows, y_p_present, metadata, window_len, output_fs
    if return_keys:
        return X_windows, y_p_present, window_len, output_fs, key_list
    return X_windows, y_p_present, window_len, output_fs
