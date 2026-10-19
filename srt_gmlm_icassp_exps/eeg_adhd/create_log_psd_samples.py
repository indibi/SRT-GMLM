import os, sys
from pathlib import Path
from collections import defaultdict
from typing import Union, Dict

import numpy as np
import pandas as pd
import torch
import mne
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, roc_auc_score,
    classification_report, confusion_matrix, roc_curve, average_precision_score,
    f1_score, roc_auc_score,
)

from srt_gmlm_icassp_exps.eeg_adhd.adhd_mne import (
    load_adhd_fif_dataset,
    EEG_CHANNELS,
    DEFAULT_SFREQ
)
from srt_gmlm_icassp_exps.eeg_adhd.create_epochs_and_psd import (
    create_sliding_window_epochs
)

def eeg_log_psd(epochs, vectorize=False):
    spectrum = epochs.compute_psd(
        picks="eeg", fmin=0.5, fmax=60.0, verbose=False
        )
    psds, freqs = spectrum.get_data(return_freqs=True)
    log_psds = np.log10(psds)
    return log_psds.reshape(len(psds), -1) if vectorize else log_psds


def load_train_val_test_split_subjects_process(
    n_sample_per_subj, fif_dir, test_ratio, val_ratio, seed=42,
    duration=15, overlap=0):
    """
    Load previously saved ADHD subjects FIF files and create Train., Val. Test sets.
    """
    adhd_dataset = load_adhd_fif_dataset(fif_dir)
    ch_names = adhd_dataset['ch_names']
    ids = adhd_dataset['ids']

    labels = np.array([adhd_dataset['subject_info'][id]['Class'] == 'ADHD' for id in ids])
    temp_subj, test_subj = train_test_split(
        ids, test_size=test_ratio, stratify=labels, random_state=seed
        )

    temp_labels = np.array(
            [adhd_dataset['subject_info'][id]['Class'] == 'ADHD' for id in temp_subj]
            )
    if val_ratio == 0:
        train_subj = temp_labels
        val_subj = np.array([])
    else:
        train_subj, val_subj = train_test_split(
            temp_subj, test_size=val_ratio, stratify=temp_labels, random_state=seed+1
            )

    event_ids = dict(ADHD=1, Control=0)
    X_st = []
    y_st = []
    X_tr = []
    y_tr = []
    X_val = []
    y_val = []

    for i, id in enumerate(ids):
        raw = adhd_dataset['raws'][id]
        event_id = adhd_dataset['subject_info'][id]['Class']
        epochs = create_sliding_window_epochs(
            raw, duration, overlap=overlap, event_id=event_ids
            )
        if len(epochs) < n_sample_per_subj:
            print((
                f"Timeseries recording for the subject {id} has {len(epochs)}"
                f" epochs < {n_sample_per_subj}."
                ))
        log_psd = eeg_log_psd(epochs[:n_sample_per_subj])
        if id in train_subj:
            X_tr.append(log_psd)
            y_tr = y_tr + [event_id == 'ADHD']*log_psd.shape[0]
        elif id in val_subj:
            X_val.append(log_psd)
            y_val = y_val + [event_id == 'ADHD']*log_psd.shape[0]
        else:
            X_st.append(log_psd)
            y_st = y_st + [event_id == 'ADHD']*log_psd.shape[0]

    X_tr = np.concat(X_tr, axis=0)
    y_tr = np.array(y_tr, dtype=np.double)
    X_val = np.concat(X_val, axis=0)
    y_val = np.array(y_val, dtype=np.double)
    X_st = np.concat(X_st, axis=0)
    y_st = np.array(y_st, dtype=np.double)
    return {
        'X_train': X_tr,
        'y_train': y_tr.reshape(-1,1),
        'X_val': X_val,
        'y_val': y_val.reshape(-1,1),
        'X_test': X_st,
        'y_test': y_st.reshape(-1,1),
    }

def standardize(X, mu=None, std=None):
    if mu is None:
        mu = np.mean(X, axis=0, keepdims=True)
    if std is None:
        std = np.std(X, axis=0, keepdims=True)
    return (X - mu)/std, mu, std

def unstandardize(X, mu, std):
    return (X*std) + mu

