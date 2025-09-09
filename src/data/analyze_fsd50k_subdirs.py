# analyze_fsd50k_final.py
# -*- coding: utf-8 -*-
"""
Performs a comprehensive 6-plot analysis of curated FSD50K audio chunks.

This script generates:
1. A single PNG file with a 2x3 grid of all plots.
2. A sub-folder named 'plots' containing each of the six plots as a separate file.
"""
import argparse
import json
from pathlib import Path
import random

import pandas as pd
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.preprocessing import MultiLabelBinarizer
from matplotlib.gridspec import GridSpec, SubplotSpec

# --- Analysis Functions ---

def analyze_chunk_files(chunk_parent_dir: Path, limit: int = None) -> pd.DataFrame:
    results = []
    audio_files = sorted(list(chunk_parent_dir.rglob("*.flac")))
    if not audio_files:
        raise FileNotFoundError(f"No .flac files found in subdirectories of {chunk_parent_dir}.")

    if limit:
        audio_files = audio_files[:limit]
    
    for audio_path in tqdm(audio_files, desc="Analyzing audio chunks"):
        try:
            original_fname_str = audio_path.parent.name
            if not original_fname_str.isdigit(): continue
            original_fname = int(original_fname_str)
            y, sr = librosa.load(audio_path, sr=None, res_type='kaiser_fast')
            results.append({
                'chunk_path': str(audio_path),
                'original_fname': original_fname,
                'duration': librosa.get_duration(y=y, sr=sr),
                'rms_energy': np.mean(librosa.feature.rms(y=y)),
                'spectral_centroid': np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)),
            })
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
    return pd.DataFrame(results)

# --- Plotting Functions ---

def plot_duration_distribution(ax: plt.Axes, data: pd.DataFrame):
    sns.histplot(ax=ax, data=data, x='duration', bins=40, kde=True)
    ax.set_title('Distribution of Chunk Durations', fontsize=16)
    ax.set_xlabel('Duration (seconds)')

def plot_label_count_distribution(ax: plt.Axes, data: pd.DataFrame):
    sns.countplot(ax=ax, data=data, x='label_count', palette='magma', hue='label_count', legend=False)
    ax.set_title('Distribution of Label Count per Chunk', fontsize=16)
    ax.set_xlabel('Number of Labels on a Single Chunk')
    ax.set_yscale('log')
    ax.set_ylabel('Count (log scale)')

def plot_top_labels(ax: plt.Axes, label_counts: pd.Series, top_n: int):
    top_labels = label_counts.head(top_n)
    sns.barplot(ax=ax, x=top_labels.values, y=top_labels.index, hue=top_labels.index, palette='viridis_r', legend=False)
    ax.set_title(f'Top {top_n} Most Common Labels', fontsize=16)
    ax.set_xlabel('Count of 10s Chunks')

def plot_co_occurrence(ax: plt.Axes, labels_series: pd.Series, top_label_names: list):
    mlb = MultiLabelBinarizer(classes=top_label_names)
    binary_matrix = mlb.fit_transform(labels_series.apply(lambda x: [l for l in x if l in top_label_names]))
    co_occurrence_matrix = (binary_matrix.T @ binary_matrix)
    np.fill_diagonal(co_occurrence_matrix, 0)
    co_occurrence_df = pd.DataFrame(co_occurrence_matrix, index=top_label_names, columns=top_label_names)
    sns.heatmap(ax=ax, data=co_occurrence_df, cmap='rocket_r', annot=False)
    ax.set_title(f'Co-occurrence of Top {len(top_label_names)} Labels', fontsize=16)

def plot_acoustic_features(ax: plt.Axes, data: pd.DataFrame):
    plot_df = data.melt(id_vars=['chunk_path'], value_vars=['rms_energy', 'spectral_centroid'])
    sns.boxenplot(ax=ax, data=plot_df, x='value', y='variable', hue='variable', palette='crest', showfliers=False, legend=False)
    ax.set_title('Acoustic Feature Distributions', fontsize=16)
    ax.set_xlabel('Value')
    ax.set_ylabel('Feature')

def plot_spectrograms(fig: plt.Figure, subplot_spec: SubplotSpec, chunk_paths: list):
    sub_gs = subplot_spec.subgridspec(3, 3, wspace=0.1, hspace=0.3)
    random_chunks = random.sample(chunk_paths, min(9, len(chunk_paths)))
    for i, chunk_path_str in enumerate(random_chunks):
        row, col = divmod(i, 3)
        ax = fig.add_subplot(sub_gs[row, col])
        y, sr = librosa.load(chunk_path_str, sr=48000)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        S_dB = librosa.power_to_db(S, ref=np.max)
        librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', ax=ax)
        ax.set_title(Path(chunk_path_str).name, fontsize=8)
        ax.set_xlabel(None); ax.set_ylabel(None)
        ax.tick_params(axis='both', which='major', labelsize=6, left=False, labelleft=False, bottom=False, labelbottom=False)

# --- Main Execution ---

def main():
    parser = argparse.ArgumentParser(description="Run a comprehensive analysis on curated FSD50K chunks.")
    parser.add_argument("chunk_parent_dir", type=Path)
    parser.add_argument("metadata_csvs", type=Path, nargs='+')
    parser.add_argument("--output_dir", type=Path, default=Path("fsd50k_analysis"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--top_n_labels", type=int, default=20)
    args = parser.parse_args()

    # --- Setup Directories ---
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = args.output_dir / "plots" # NEW: Define plots subdirectory
    plots_dir.mkdir(parents=True, exist_ok=True)
    analysis_name = args.chunk_parent_dir.name
    print(f"Starting comprehensive analysis on: {analysis_name}")
    
    # --- Load and Process Data ---
    all_dfs = [pd.read_csv(p) for p in args.metadata_csvs if p.exists()]
    if not all_dfs: print("Error: No valid metadata files loaded."); return
    meta_df = pd.concat(all_dfs, ignore_index=True).set_index('fname')
    
    analysis_df = analyze_chunk_files(args.chunk_parent_dir, args.limit)
    if analysis_df.empty: print("Analysis finished early: No audio files processed."); return
        
    merged_df = analysis_df.join(meta_df, on='original_fname').dropna(subset=['labels'])
    labels_series = merged_df['labels'].str.split(',')
    label_counts = labels_series.explode().value_counts()
    merged_df['label_count'] = labels_series.apply(len)
    top_label_names = label_counts.head(args.top_n_labels).index.tolist()

    # --- Generate and Save Individual Plots ---
    print("Generating and saving individual plots...")
    plot_functions = {
        "01_duration_distribution.png": (plot_duration_distribution, {'data': merged_df}),
        "02_label_count_distribution.png": (plot_label_count_distribution, {'data': merged_df}),
        "03_top_labels.png": (plot_top_labels, {'label_counts': label_counts, 'top_n': args.top_n_labels}),
        "04_co_occurrence_heatmap.png": (plot_co_occurrence, {'labels_series': labels_series, 'top_label_names': top_label_names}),
        "05_acoustic_features.png": (plot_acoustic_features, {'data': merged_df})
    }
    for filename, (func, params) in plot_functions.items():
        fig_single, ax_single = plt.subplots(1, 1, figsize=(10, 8))
        func(ax_single, **params)
        fig_single.tight_layout()
        fig_single.savefig(plots_dir / filename, dpi=150)
        plt.close(fig_single)

    # --- Generate and Save Comprehensive Plot ---
    print("Generating comprehensive 6-plot report...")
    fig, axes = plt.subplots(2, 3, figsize=(24, 14))
    fig.suptitle(f'Comprehensive Analysis: "{analysis_name}"', fontsize=22, y=0.98)

    plot_duration_distribution(axes[0, 0], merged_df)
    plot_label_count_distribution(axes[0, 1], merged_df)
    plot_top_labels(axes[0, 2], label_counts, args.top_n_labels)
    plot_co_occurrence(axes[1, 0], labels_series, top_label_names)
    plot_acoustic_features(axes[1, 1], merged_df)
    
    # Handle the spectrogram grid
    axes[1, 2].set_title('6. Example Mel-Spectrograms', fontsize=16)
    axes[1, 2].axis('off')
    plot_spectrograms(fig, axes[1, 2].get_gridspec()[1, 2], merged_df['chunk_path'].tolist())
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plot_path = args.output_dir / f"comprehensive_analysis_{analysis_name}.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"\nComprehensive analysis plot saved to {plot_path}")
    print(f"Individual plots saved in: {plots_dir}")

if __name__ == '__main__':
    main()