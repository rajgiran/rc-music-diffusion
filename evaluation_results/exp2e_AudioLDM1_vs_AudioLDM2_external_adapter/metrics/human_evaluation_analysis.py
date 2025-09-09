
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_human_evaluation(results_file):
    """
    Analyze human evaluation results
    Expected columns: pair_id, criterion, preference, confidence, listener_id
    """
    df = pd.read_csv(results_file)
    df['preference_numeric'] = df['preference'].map({'A': 0, 'B': 1})

    overall = {}
    for criterion in df['criterion'].unique():
        d = df[df['criterion'] == criterion]
        counts = d['preference'].value_counts()
        total = len(d)
        successes = counts.get('B', 0)  # B == model 2
        p_value = stats.binom_test(successes, total, p=0.5, alternative='two-sided')
        prop = successes / max(total, 1)
        me = 1.96 * np.sqrt(prop * (1 - prop) / max(total, 1))
        overall[criterion] = {
            'pref_A_percent': 100.0 * counts.get('A', 0) / max(total, 1),
            'pref_B_percent': 100.0 * counts.get('B', 0) / max(total, 1),
            'p_value': float(p_value),
            'significant': bool(p_value < 0.05),
            'conf_int': (float(prop - me), float(prop + me)),
            'n': int(total),
        }
    return overall

def calculate_inter_rater_reliability(df):
    """Mean pairwise Pearson correlation as a simple reliability proxy."""
    from scipy.stats import pearsonr
    pivot = df.pivot_table(index='pair_id', columns='listener_id',
                           values='preference_numeric', aggfunc='first')
    corrs = []
    cols = list(pivot.columns)
    for i in range(len(cols)):
        for j in range(i+1, len(cols)):
            valid = ~(pivot[cols[i]].isna() | pivot[cols[j]].isna())
            if valid.sum() > 0:
                c, _ = pearsonr(pivot.loc[valid, cols[i]], pivot.loc[valid, cols[j]])
                corrs.append(c)
    return float(np.mean(corrs)) if corrs else 0.0
