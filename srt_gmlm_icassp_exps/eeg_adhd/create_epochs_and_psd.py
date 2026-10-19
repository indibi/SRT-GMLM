"""
Sliding-Window Epoching and Sensor Power Spectral Density (PSD) Feature Extraction for ADHD EEG.

Mirrors the MNE sleep_stage_classification feature extraction workflow:
- Creates sliding-window epochs from Raw objects.
- Extracts relative band power features across 19 standard 10-20 EEG channels.
- Formats data into both vectorized feature matrices and 3D tensor representations.
"""

import os, sys
from pathlib import Path

# Ensure project root is on sys.path
project_root = str(Path(__file__).resolve().parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from typing import Dict, List, Optional, Tuple, Union
import argparse
import time

import numpy as np
import mne

from srt_gmlm_icassp_exps.eeg_adhd.adhd_mne import (
    EEG_CHANNELS,
    DEFAULT_SFREQ,
    load_adhd_fif_dataset
)


# Standard clinical EEG frequency bands
DEFAULT_FREQ_BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}


def create_sliding_window_epochs(
    raw: mne.io.BaseRaw,
    duration: float = 4.0,
    overlap: float = 2.0,
    event_id: Optional[Dict[str, int]] = None,
    verbose: bool = False,
) -> mne.Epochs:
    """
    Divide a continuous MNE Raw object into sliding-window Epochs.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Continuous EEG Raw object.
    duration : float, default 4.0
        Duration of each epoch in seconds.
    overlap : float, default 2.0
        Overlap between consecutive epochs in seconds.
    event_id : dict, optional
        Mapping of event descriptions to integer IDs (e.g. {'ADHD': 1} or {'Control': 0}).
        If None, infer from raw.info['description'].
    verbose : bool, default False
        MNE verbosity flag.

    Returns
    -------
    mne.Epochs
        Fixed-length sliding window epochs with event labels and metadata.
    """
    if event_id is None:
        desc = raw.info.get("description", "")
        if "ADHD" in desc:
            class_name = "ADHD"
            event_code = 1
        elif "Control" in desc:
            class_name = "Control"
            event_code = 0
        else:
            class_name = "Unknown"
            event_code = 1
        event_dict = {class_name: event_code}
    else:
        event_dict = event_id
        event_code = list(event_dict.values())[0]
    epochs = mne.make_fixed_length_epochs(
        raw,
        duration=duration,
        overlap=overlap,
        id=event_code,
        preload=True,
        verbose=verbose,
    )
    
    epochs.event_id = event_dict
    return epochs


def eeg_power_band(
    epochs: mne.Epochs,
    freq_bands: Optional[Dict[str, Tuple[float, float]]] = None,
    relative: bool = True,
    vectorize: bool = True,
    fmin: float = 0.5,
    fmax: float = 45.0,
    verbose: bool = False,
) -> np.ndarray:
    """
    EEG relative power band feature extraction.

    Directly compatible with scikit-learn Pipeline / FunctionTransformer,
    matching the implementation in `sleep_stage_classification.ipynb`.

    Parameters
    ----------
    epochs : mne.Epochs
        The EEG epochs.
    freq_bands : dict, optional
        Dictionary mapping band names to (fmin, fmax) tuples.
        Defaults to Delta (0.5-4), Theta (4-8), Alpha (8-13), Beta (13-30), Gamma (30-45).
    relative : bool, default True
        If True, normalize PSD across frequencies to obtain relative power (summing to 1).
    vectorize : bool, default True
        If True, flatten features to shape [n_epochs, n_channels * n_bands].
        If False, return 3D tensor of shape [n_epochs, n_channels, n_bands].
    fmin : float, default 0.5
        Lower frequency bound for PSD estimation.
    fmax : float, default 45.0
        Upper frequency bound for PSD estimation.
    verbose : bool, default False
        MNE verbosity flag.

    Returns
    -------
    X : np.ndarray
        Transformed feature array of shape [n_epochs, n_channels * n_bands] (if vectorize=True)
        or [n_epochs, n_channels, n_bands] (if vectorize=False).
    """
    if freq_bands is None:
        freq_bands = DEFAULT_FREQ_BANDS

    # Compute Power Spectral Density via Welch's method across EEG channels
    spectrum = epochs.compute_psd(
        picks="eeg",
        fmin=fmin,
        fmax=fmax,
        verbose=verbose,
    )
    psds, freqs = spectrum.get_data(return_freqs=True)  # (n_epochs, n_channels, n_freqs)

    # Normalize PSDs across frequency axis to obtain relative power
    if relative:
        denom = np.sum(psds, axis=-1, keepdims=True)
        denom[denom == 0] = 1.0  # avoid division by zero
        psds = psds / denom

    band_features = []
    for (f_low, f_high) in freq_bands.values():
        mask = (freqs >= f_low) & (freqs < f_high)
        # Mean spectral power within the frequency band: shape (n_epochs, n_channels)
        band_power = psds[:, :, mask].mean(axis=-1, keepdims=not vectorize)
        if vectorize:
            band_features.append(band_power.reshape(len(psds), -1))
        else:
            band_features.append(band_power)

    # Concatenate across band dimension
    if vectorize:
        return np.concatenate(band_features, axis=1)
    else:
        return np.concatenate(band_features, axis=2)


def extract_and_save_dataset_features(
    fif_dir: Union[str, Path],
    output_file: Optional[Union[str, Path]] = None,
    duration: float = 4.0,
    overlap: float = 2.0,
    freq_bands: Optional[Dict[str, Tuple[float, float]]] = None,
    verbose: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Extract sliding-window epochs and PSD features for all 121 subjects.

    Computes:
    - Epoch-level features and labels with subject grouping IDs.
    - Subject-level aggregated features and labels.
    - Feature names mapping channels and frequency bands.

    Parameters
    ----------
    fif_dir : str or Path
        Directory containing `<subj_id>_raw.fif` files.
    output_file : str or Path, optional
        File path to save the extracted `.npz` archive.
    duration : float, default 4.0
        Sliding-window duration in seconds.
    overlap : float, default 2.0
        Sliding-window overlap in seconds.
    freq_bands : dict, optional
        Frequency bands definition.
    verbose : bool, default False
        MNE verbosity flag.

    Returns
    -------
    dict
        Dictionary with:
        - 'X_epochs': (N_epochs, n_features)
        - 'y_epochs': (N_epochs,)
        - 'groups_epochs': (N_epochs,) subject ID string for each epoch
        - 'X_tensor': (N_epochs, n_channels, n_bands)
        - 'X_subject': (121, n_features) subject-averaged PSD features
        - 'y_subject': (121,) subject-level binary labels (1=ADHD, 0=Control)
        - 'subject_ids': (121,) array of subject ID strings
        - 'feature_names': (n_features,) names (e.g. 'Fp1_delta', 'Fp1_theta', ...)
        - 'channel_names': list of 19 channels
        - 'band_names': list of frequency band names
    """
    if freq_bands is None:
        freq_bands = DEFAULT_FREQ_BANDS

    fif_dir = Path(fif_dir)
    fif_files = sorted(fif_dir.glob("*_raw.fif"))
    if not fif_files:
        raise FileNotFoundError(f"No *_raw.fif files found in {fif_dir}")

    print(f"Extracting sliding-window epochs (duration={duration}s, overlap={overlap}s) for {len(fif_files)} subjects...")
    t0 = time.time()

    all_X_epochs = []
    all_X_tensor = []
    all_y_epochs = []
    all_groups_epochs = []

    subject_ids = []
    all_X_subject = []
    all_y_subject = []

    ch_names = None
    band_names = list(freq_bands.keys())

    for idx, fif_path in enumerate(fif_files, start=1):
        subj_id = fif_path.name.replace("_raw.fif", "")
        raw = mne.io.read_raw_fif(fif_path, preload=True, verbose=False)

        if ch_names is None:
            ch_names = raw.ch_names

        desc = raw.info.get("description", "")
        is_adhd = 1 if "ADHD" in desc else 0

        # Create sliding window epochs
        epochs = create_sliding_window_epochs(
            raw,
            duration=duration,
            overlap=overlap,
            event_id={"ADHD": 1} if is_adhd == 1 else {"Control": 0},
            verbose=False,
        )

        n_epochs = len(epochs)
        if n_epochs == 0:
            print(f"Warning: Subject {subj_id} yielded 0 epochs (duration={raw.times[-1]:.2f}s). Skipping.")
            continue

        # Extract vectorized PSD features: (n_epochs, n_channels * n_bands)
        X_vec = eeg_power_band(epochs, freq_bands=freq_bands, relative=True, vectorize=True, verbose=False)
        # Extract 3D tensor features: (n_epochs, n_channels, n_bands)
        X_ten = eeg_power_band(epochs, freq_bands=freq_bands, relative=True, vectorize=False, verbose=False)

        all_X_epochs.append(X_vec)
        all_X_tensor.append(X_ten)
        all_y_epochs.append(np.full(n_epochs, is_adhd, dtype=int))
        all_groups_epochs.extend([subj_id] * n_epochs)

        # Subject-level feature: mean PSD band power across all epochs of this subject
        all_X_subject.append(X_vec.mean(axis=0))
        all_y_subject.append(is_adhd)
        subject_ids.append(subj_id)

        if idx % 25 == 0 or idx == len(fif_files):
            print(f"  Processed {idx}/{len(fif_files)} subjects ({len(all_groups_epochs)} total epochs generated)")

    X_epochs = np.concatenate(all_X_epochs, axis=0).astype(np.float32)
    X_tensor = np.concatenate(all_X_tensor, axis=0).astype(np.float32)
    y_epochs = np.concatenate(all_y_epochs, axis=0).astype(int)
    groups_epochs = np.array(all_groups_epochs, dtype=str)

    X_subject = np.stack(all_X_subject, axis=0).astype(np.float32)
    y_subject = np.array(all_y_subject, dtype=int)
    subject_ids = np.array(subject_ids, dtype=str)

    # Feature names: (n_channels * n_bands)
    feature_names = []
    for band in band_names:
        for ch in ch_names:
            feature_names.append(f"{ch}_{band}")
    feature_names = np.array(feature_names, dtype=str)

    elapsed = time.time() - t0
    print(f"\nFeature Extraction Complete in {elapsed:.2f}s:")
    print(f"- Total Epochs: {X_epochs.shape[0]} across {len(subject_ids)} subjects")
    print(f"- Features per epoch: {X_epochs.shape[1]} (19 channels x {len(band_names)} bands)")
    print(f"- Epoch Class Balance: {np.sum(y_epochs == 1)} ADHD vs {np.sum(y_epochs == 0)} Control")
    print(f"- Subject Class Balance: {np.sum(y_subject == 1)} ADHD vs {np.sum(y_subject == 0)} Control")

    result = {
        "X_epochs": X_epochs,
        "y_epochs": y_epochs,
        "groups_epochs": groups_epochs,
        "X_tensor": X_tensor,
        "X_subject": X_subject,
        "y_subject": y_subject,
        "subject_ids": subject_ids,
        "feature_names": feature_names,
        "channel_names": np.array(ch_names, dtype=str),
        "band_names": np.array(band_names, dtype=str),
        "duration": np.float32(duration),
        "overlap": np.float32(overlap),
    }

    if output_file is not None:
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output_file, **result)
        print(f"- Saved feature archive to: {output_file} ({output_file.stat().st_size / 1024 / 1024:.2f} MB)")

    return result


def main():
    parser = argparse.ArgumentParser(description="Create sliding-window epochs and extract PSD features for ADHD dataset.")
    default_fif_dir = Path(__file__).resolve().parent.parent.parent / "eeg_dataset_for_adhd" / "raw_fif"
    default_out = Path(__file__).resolve().parent.parent.parent / "eeg_dataset_for_adhd" / "adhd_psd_features.npz"

    parser.add_argument("--fif-dir", type=str, default=str(default_fif_dir), help="Directory of subject *_raw.fif files")
    parser.add_argument("--output", type=str, default=str(default_out), help="Output .npz file path")
    parser.add_argument("--duration", type=float, default=4.0, help="Epoch duration in seconds (default: 4.0)")
    parser.add_argument("--overlap", type=float, default=2.0, help="Epoch overlap in seconds (default: 2.0)")

    args = parser.parse_args()

    extract_and_save_dataset_features(
        fif_dir=args.fif_dir,
        output_file=args.output,
        duration=args.duration,
        overlap=args.overlap,
    )


if __name__ == "__main__":
    main()
