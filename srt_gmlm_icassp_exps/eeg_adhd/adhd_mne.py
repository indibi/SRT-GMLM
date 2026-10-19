"""
EEG Data Processing for ADHD Dataset into MNE Raw Format.

Processes EEG recordings from 121 subjects in `adhdata.csv` (19 EEG channels,
128 Hz sampling frequency, ADHD vs Control) into MNE `Raw` (`mne.io.RawArray`) objects.
"""

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import argparse
import sys

import numpy as np
import pandas as pd
import mne


EEG_CHANNELS = [
    'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4',
    'O1', 'O2', 'F7', 'F8', 'T7', 'T8', 'P7', 'P8',
    'Fz', 'Cz', 'Pz',
]

DEFAULT_SFREQ = 128.0
DEFAULT_SCALE = 1e-6  # Convert microvolts (uV) to Volts (V) for MNE standard units
DEFAULT_MONTAGE = 'standard_1020'


def create_subject_raw(
    subj_df: pd.DataFrame,
    subj_id: str,
    subj_class: str,
    channels: Optional[List[str]] = None,
    sfreq: float = DEFAULT_SFREQ,
    scale: float = DEFAULT_SCALE,
    montage_name: Optional[str] = DEFAULT_MONTAGE,
    verbose: bool = False,
) -> mne.io.RawArray:
    """
    Convert a single subject's EEG DataFrame into an MNE RawArray.

    Parameters
    ----------
    subj_df : pd.DataFrame
        DataFrame containing EEG channel columns for one subject.
    subj_id : str
        Subject identifier (e.g. 'v10p').
    subj_class : str
        Subject class label ('ADHD' or 'Control').
    channels : list of str, optional
        List of channel names. Defaults to the 19 standard 10-20 channels.
    sfreq : float, default 128.0
        Sampling frequency in Hz.
    scale : float, default 1e-6
        Multiplier to convert data to Volts (MNE standard unit is Volts;
        default 1e-6 converts microvolts to Volts). Use 1.0 if raw unscaled values are desired.
    montage_name : str, optional, default 'standard_1020'
        Standard montage name to apply.
    verbose : bool, default False
        MNE verbosity flag.

    Returns
    -------
    mne.io.RawArray
        MNE RawArray object with info, subject metadata, and montage attached.
    """
    if channels is None:
        channels = [ch for ch in EEG_CHANNELS if ch in subj_df.columns]

    # Extract data as (n_channels, n_times)
    data = subj_df[channels].to_numpy().T.astype(np.float64)
    if scale != 1.0:
        data = data * scale

    # Create MNE Info
    info = mne.create_info(ch_names=channels, sfreq=sfreq, ch_types='eeg', verbose=verbose)

    # Attach subject metadata
    info['subject_info'] = {'his_id': str(subj_id)}
    info['description'] = f'Class: {subj_class}'

    # Create RawArray
    raw = mne.io.RawArray(data, info, verbose=verbose)

    # Apply electrode montage if specified
    if montage_name is not None:
        montage = mne.channels.make_standard_montage(montage_name)
        raw.set_montage(montage, verbose=verbose)

    return raw


def parse_adhdata_to_mne(
    csv_path: Union[str, Path],
    channels: Optional[List[str]] = None,
    sfreq: float = DEFAULT_SFREQ,
    scale: float = DEFAULT_SCALE,
    montage_name: Optional[str] = DEFAULT_MONTAGE,
    save_fif_dir: Optional[Union[str, Path]] = None,
    overwrite_fif: bool = True,
    verbose: bool = False,
) -> Dict:
    """
    Parse adhdata.csv containing 121 subjects into MNE RawArray objects.

    Parameters
    ----------
    csv_path : str or Path
        Path to `adhdata.csv`.
    channels : list of str, optional
        List of channel names. Defaults to standard 19 EEG channels.
    sfreq : float, default 128.0
        Sampling frequency in Hz.
    scale : float, default 1e-6
        Scale factor to convert to Volts (uV -> V).
    montage_name : str, optional, default 'standard_1020'
        Montage name to set.
    save_fif_dir : str or Path, optional
        Directory to save individual `<subject_id>_raw.fif` files. If None, files are not saved.
    overwrite_fif : bool, default True
        Whether to overwrite existing .fif files if save_fif_dir is specified.
    verbose : bool, default False
        MNE verbosity flag.

    Returns
    -------
    dict
        Dictionary containing:
        - 'raws': dict mapping {subj_id: mne.io.RawArray}
        - 'ids': list of all subject IDs
        - 'adhd_ids': list of ADHD subject IDs
        - 'control_ids': list of Control subject IDs
        - 'subject_info': dict mapping {subj_id: {'class': ..., 'n_samples': ..., 'duration_sec': ...}}
        - 'ch_names': list of channel names
        - 'sfreq': float sampling frequency
        - 'fif_paths': dict mapping {subj_id: Path} (if save_fif_dir is provided)
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset CSV not found at: {csv_path}")

    if channels is None:
        channels = list(EEG_CHANNELS)

    print(f"Reading CSV from {csv_path}...")
    df = pd.read_csv(csv_path)

    if save_fif_dir is not None:
        save_fif_dir = Path(save_fif_dir)
        save_fif_dir.mkdir(parents=True, exist_ok=True)

    raws = {}
    subject_info = {}
    adhd_ids = []
    control_ids = []
    fif_paths = {}

    # Subjects are contiguous blocks in adhdata.csv
    # Using groupby preserves ordering if observed
    grouped = df.groupby('ID', sort=False)
    total_subjects = len(grouped)
    print(f"Found {total_subjects} subjects in dataset. Processing into MNE Raw objects...")

    for i, (subj_id, subj_df) in enumerate(grouped, start=1):
        subj_class = subj_df['Class'].iloc[0]
        n_samples = len(subj_df)
        duration_sec = n_samples / sfreq

        raw = create_subject_raw(
            subj_df=subj_df,
            subj_id=subj_id,
            subj_class=subj_class,
            channels=channels,
            sfreq=sfreq,
            scale=scale,
            montage_name=montage_name,
            verbose=verbose,
        )

        raws[subj_id] = raw
        subject_info[subj_id] = {
            'ID': subj_id,
            'Class': subj_class,
            'n_samples': n_samples,
            'duration_sec': duration_sec,
        }

        if subj_class == 'ADHD':
            adhd_ids.append(subj_id)
        elif subj_class == 'Control':
            control_ids.append(subj_id)
        else:
            print(f"Warning: Unknown class '{subj_class}' for subject '{subj_id}'")

        if save_fif_dir is not None:
            fif_fname = save_fif_dir / f"{subj_id}_raw.fif"
            raw.save(fif_fname, overwrite=overwrite_fif, verbose=verbose)
            fif_paths[subj_id] = fif_fname

        if i % 20 == 0 or i == total_subjects:
            print(f"  Processed {i}/{total_subjects} subjects ({len(adhd_ids)} ADHD, {len(control_ids)} Control)")

    print("Finished processing all subjects.")

    result = {
        'raws': raws,
        'ids': list(raws.keys()),
        'adhd_ids': adhd_ids,
        'control_ids': control_ids,
        'subject_info': subject_info,
        'ch_names': channels,
        'sfreq': sfreq,
    }
    if save_fif_dir is not None:
        result['fif_paths'] = fif_paths

    return result


def load_adhd_fif_dataset(
    fif_dir: Union[str, Path],
    preload: bool = False,
    verbose: bool = False,
) -> Dict:
    """
    Load previously saved ADHD subject FIF files from a directory.

    Parameters
    ----------
    fif_dir : str or Path
        Directory containing `<subj_id>_raw.fif` files.
    preload : bool, default False
        Whether to preload raw data into memory.
    verbose : bool, default False
        MNE verbosity flag.

    Returns
    -------
    dict
        Dictionary with 'raws', 'ids', 'adhd_ids', 'control_ids', 'subject_info'.
    """
    fif_dir = Path(fif_dir)
    fif_files = sorted(fif_dir.glob("*_raw.fif"))

    if not fif_files:
        raise FileNotFoundError(f"No '*_raw.fif' files found in {fif_dir}")

    raws = {}
    adhd_ids = []
    control_ids = []
    subject_info = {}

    for fif_path in fif_files:
        subj_id = fif_path.name.replace("_raw.fif", "")
        raw = mne.io.read_raw_fif(fif_path, preload=preload, verbose=verbose)
        raws[subj_id] = raw

        desc = raw.info.get('description', '')
        subj_class = 'ADHD' if 'ADHD' in desc else ('Control' if 'Control' in desc else 'Unknown')
        if subj_class == 'ADHD':
            adhd_ids.append(subj_id)
        elif subj_class == 'Control':
            control_ids.append(subj_id)

        subject_info[subj_id] = {
            'ID': subj_id,
            'Class': subj_class,
            'n_samples': len(raw),
            'duration_sec': raw.times[-1],
        }

    return {
        'raws': raws,
        'ids': list(raws.keys()),
        'adhd_ids': adhd_ids,
        'control_ids': control_ids,
        'subject_info': subject_info,
        'ch_names': raws[list(raws.keys())[0]].ch_names,
        'sfreq': raws[list(raws.keys())[0]].info['sfreq'],
    }


def main():
    parser = argparse.ArgumentParser(description="Process ADHD EEG dataset into MNE Raw format.")
    default_csv = Path(__file__).resolve().parent.parent.parent / 'data' / 'eeg_adhd'/ 'adhdata.csv'
    default_out = Path(__file__).resolve().parent.parent.parent / 'data' / 'eeg_adhd'/ 'raw_fif'

    parser.add_argument('--csv-path', type=str, default=str(default_csv), help="Path to adhdata.csv")
    parser.add_argument('--save-fif', action='store_true', help="Save individual Raw FIF files")
    parser.add_argument('--output-dir', type=str, default=str(default_out), help="Output directory for FIF files")
    parser.add_argument('--scale', type=float, default=DEFAULT_SCALE, help="Scale factor to Volts (default: 1e-6)")
    parser.add_argument('--sfreq', type=float, default=DEFAULT_SFREQ, help="Sampling frequency (default: 128.0)")

    args = parser.parse_args()

    save_dir = args.output_dir if args.save_fif else None
    dataset = parse_adhdata_to_mne(
        csv_path=args.csv_path,
        sfreq=args.sfreq,
        scale=args.scale,
        save_fif_dir=save_dir,
    )

    print(f"\nProcessing Summary:")
    print(f"- Total subjects: {len(dataset['raws'])}")
    print(f"- ADHD subjects: {len(dataset['adhd_ids'])}")
    print(f"- Control subjects: {len(dataset['control_ids'])}")
    print(f"- Channels ({len(dataset['ch_names'])}): {dataset['ch_names']}")
    print(f"- Sampling rate: {dataset['sfreq']} Hz")
    if save_dir:
        print(f"- Saved FIF files to: {save_dir}")


if __name__ == '__main__':
    main()
