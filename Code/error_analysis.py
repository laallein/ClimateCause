import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar
from scipy.stats import chi2_contingency
import numpy as np
from scipy.stats import kruskal, rankdata, norm

def mcnemar_test(f1: str,
                 f2: str
                 ):
    df1 = pd.read_excel(f1, engine='openpyxl')
    df2 = pd.read_excel(f2, engine='openpyxl')

    # Identify ground truth and prediction columns
    if "CorrI" in f1:
        ground_truth_col = 'Correlation (positive/negative or increase-increase/increase-decrease)'
    else:
        ground_truth_col = 'chain_membership_label'

    prediction_cols_1 = [col for col in df1.columns if col.endswith('_labels')]
    prediction_cols_2 = [col for col in df2.columns if col.endswith('_labels')]

    results = []

    # Compute McNemar's test for each pair of prediction columns
    for col_1, col_2 in zip(prediction_cols_1, prediction_cols_2):
        table = pd.crosstab(df1[col_1], df2[col_2])
        result = mcnemar(table, exact=False)

        alpha = 0.05
        adjusted_alpha = bonferroni_correction(alpha, 9)
        significance = True if result.pvalue < adjusted_alpha else False

        results.append({
            'Model A': col_1,
            'Model B': col_2,
            'Statistic': result.statistic,
            'p-value': result.pvalue,
            'significance': significance
        })

    # Create summary DataFrame
    summary_df = pd.DataFrame(results)
    print(summary_df)

    for index, row in summary_df.iterrows():
        print(f"Row {index}: {row.to_dict()}")

    # Optionally save to CSV
    # summary_df.to_csv('mcnemar_results.csv', index=False)

def bonferroni_correction(a, l):
    return a / l


def add_majority_vote_column(df, new_col_name="majority_vote"):
    # Convert values to numeric, treating missing values as NaN
    columns = [col for col in df.columns if col.endswith('_labels')]
    df[columns] = df[columns].apply(pd.to_numeric, errors='coerce')

    # Compute majority vote row-wise
    def majority_vote(row):
        values = row.dropna().tolist()
        if len(values) == 0:
            return np.nan  # No votes available
        return 1 if values.count(1) >= values.count(0) else 0

    df[new_col_name] = df[columns].apply(majority_vote, axis=1)
    return df

def chi_squared_test_with_correction(results_file: str):
    df = pd.read_excel(results_file, engine='openpyxl')
    if "CorrI" in results_file:
        gt_col = 'Correlation (positive/negative or increase-increase/increase-decrease)'
    else:
        gt_col = 'chain_membership_label'

    # Identify categorical columns
    categorical_cols = ['Explicit/implicit', "Causation: Positive/Negative (causes/doesn't cause or produce/do_not_produce)", "Combined", "Nested causality"]
    prediction_cols = [col for col in df.columns if col.endswith('_labels')]

    print("Independence between variables and predictions...")
    for col in categorical_cols:
        combis = [(col_, col) for col_ in prediction_cols]
        chi_test(combis, col, df, gt_col)

    print("Independence between variables and incorrect predictions...")
    correct_cols = []
    for prediction_col in prediction_cols:
        col_name = prediction_col + "_CORRECT"
        correct_cols.append(col_name)
        df[col_name] = (df['Correlation (positive/negative or increase-increase/increase-decrease)'] == df[prediction_col]).astype(int)
    for col in categorical_cols:
        combis = [(col, col_) for col_ in correct_cols]
        chi_test(combis, col, df, gt_col)


def chi_test(c, v, df, ground_truth_column):
    results = []
    # Perform Chi-squared test for all pairs
    for col_a, col_b in c:
        contingency_table = pd.crosstab(df[col_a], df[col_b])
        chi2, p, dof, expected = chi2_contingency(contingency_table, correction=True)
        results.append({
            'Column A': col_a,
            'Column B': col_b,
            'Chi2 Statistic': chi2,
            'p-value': p
        })

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Add significance flag
    alpha = 0.05
    results_df['Significant'] = results_df['p-value'] < alpha

    #true_count = results_df['Significant (Bonferroni)'].sum()
    true_count = results_df['Significant'].sum()
    print(
        f"{v}: Number of significant results with Yates correction: {true_count} / {len(results_df['Significant'])}")

    col_at_inspection = c[0][1]
    contingency_table = pd.crosstab(df[ground_truth_column], df[col_at_inspection])
    chi2, p, dof, expected = chi2_contingency(contingency_table, correction=True)


def chi_test_bonferroni(c, v, df):
    results = []

    # Perform Chi-squared test for all pairs
    for col_a, col_b in c:
        contingency_table = pd.crosstab(df[col_a], df[col_b])
        #print(contingency_table)
        chi2, p, dof, expected = chi2_contingency(contingency_table, correction=True)
        #print(chi2, p, dof, expected)
        results.append({
            'Column A': col_a,
            'Column B': col_b,
            'Chi2 Statistic': chi2,
            'p-value': p
        })

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Add significance flag
    alpha = 0.05
    results_df['Significant'] = results_df['p-value'] < alpha

    true_count = results_df['Significant'].sum()
    print(
        f"{v}: Number of significant results with Yates correction: {true_count} / {len(results_df['Significant'])}")


def per_class_test(f, true_col='chain_position_label', class_labels=(0, 1, 2, 3), print_confusion=True, normalize='none'):
    df = pd.read_excel(f, engine='openpyxl')
    # Extract true and predicted labels
    prediction_cols = [col for col in df.columns if col.endswith('_labels')]

    y_true = df[true_col].values

    rows = []
    for pred_col in prediction_cols:
        y_pred = df[pred_col].values

        for c in class_labels:
            # One-vs-rest for class c
            tp = int(((y_pred == c) & (y_true == c)).sum())
            fp = int(((y_pred == c) & (y_true != c)).sum())
            fn = int(((y_pred != c) & (y_true == c)).sum())
            support = int((y_true == c).sum())

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

            rows.append({
                'prediction_col': pred_col,
                'class': c,
                'support': support,
                'tp': tp,
                'fp': fp,
                'fn': fn,
                'precision': precision,
                'recall': recall,
                'f1': f1
            })

            # ---- Confusion matrix (printed) ----
            if print_confusion:
                cm = pd.crosstab(
                    pd.Categorical(df[true_col], categories=class_labels, ordered=True),
                    pd.Categorical(df[pred_col], categories=class_labels, ordered=True),
                    rownames=['True'],
                    colnames=['Pred'],
                    dropna=False
                ).astype(int)

                if normalize == 'row':
                    denom = cm.sum(axis=1).replace(0, 1)
                    cm_print = (cm.T / denom).T
                elif normalize == 'col':
                    denom = cm.sum(axis=0).replace(0, 1)
                    cm_print = cm / denom
                else:
                    cm_print = cm

                # Nice header
                title_suffix = {
                    'none': 'counts',
                    'row': 'row-normalized',
                    'col': 'column-normalized'
                }[normalize]
                print(f"\n=== Confusion Matrix ({title_suffix}) — {pred_col} ===")
                print(cm_print.to_string())

    results_df = pd.DataFrame(rows)
    # Order columns for readability
    results_df = results_df[
        ['prediction_col', 'class', 'support', 'tp', 'fp', 'fn', 'precision', 'recall', 'f1']
    ].reset_index(drop=True)

    return results_df


def summarize_per_class(results_df):
    # Keep only actual class rows
    mask_classes = results_df['class'].isin([0, 1, 2, 3])
    per_class = results_df.loc[mask_classes].copy()

    # Group by class across all prediction columns (models)
    agg = (
        per_class
        .groupby('class')
        .agg(
            n_models=('prediction_col', 'nunique'),
            n=('support', 'mean'),
            mean_precision=('precision', 'mean'),
            std_precision=('precision', 'std'),
            mean_recall=('recall', 'mean'),
            std_recall=('recall', 'std'),
            mean_f1=('f1', 'mean'),
            std_f1=('f1', 'std')
        )
        .reset_index()
        .sort_values('class')
    )

    # Optional: round for readability
    return agg.round(4)

def association_with_complexity_metrics(f):
    complexity_df = pd.read_csv("complexity_results.csv")
    ccr_df = pd.read_excel(f, engine='openpyxl')

    df = pd.merge(ccr_df, complexity_df, left_on='ipccText', right_on='statement', how='inner')
    model_predictions = [col for col in df.columns if col.endswith('_labels')]
    complexity_labels = [
                         'total_complexity'
    ]

    if "position" in f:
        gt_col = 'chain_position_label'
    else:
        gt_col = 'chain_membership_label'

    for model in model_predictions:
        combis = [(col_, model) for col_ in complexity_labels]
        kruskal_test(combis, df, p_adjust='holm')


def kruskal_effect_and_dunn(df, continuous_col, categorical_col, p_adjust='holm'):
    """
    Run Kruskal–Wallis test for a continuous variable across categories,
    compute epsilon-squared effect size, and perform Dunn's post-hoc tests.

    Parameters
    ----------
    df : pd.DataFrame
    continuous_col : str
        Column name of the continuous variable.
    categorical_col : str
        Column name of the categorical labels (nominal/ordinal).
    p_adjust : {'bonferroni', 'holm', 'none'}
        Multiple-comparison adjustment for Dunn's post-hoc.

    Returns
    -------
    result : dict
        {
          'kruskal': {'H': float, 'p': float, 'epsilon2': float, 'k': int, 'n': int},
          'dunn': pd.DataFrame with columns ['group_i','group_j','z','p_raw','p_adj']
        }
    """
    # Drop NaNs on the relevant columns
    data = df[[continuous_col, categorical_col]].dropna()
    y = data[continuous_col].values
    g = data[categorical_col].values

    # Build groups
    groups = []
    levels = []
    for lvl, grp in data.groupby(categorical_col):
        arr = grp[continuous_col].values
        if len(arr) > 0:
            groups.append(arr)
            levels.append(lvl)

    k = len(groups)
    n = sum(len(g_) for g_ in groups)

    if k < 2:
        raise ValueError(f"Need at least 2 groups in '{categorical_col}'. Got {k}.")

    # --- Kruskal–Wallis test ---
    H, p_kw = kruskal(*groups)

    # Epsilon-squared effect size (for Kruskal–Wallis)
    # Common field formula: epsilon^2 = (H - k + 1) / (n - k)
    epsilon2 = (H - k + 1) / (n - k) if (n - k) > 0 else np.nan

    # --- Dunn's post-hoc test ---
    # Rank all observations
    ranks = rankdata(y, method='average')
    # Map ranks back to groups
    data = data.assign(_rank=ranks)
    group_ranks = [data.loc[data[categorical_col] == lvl, '_rank'].values for lvl in levels]
    group_ns = [len(r) for r in group_ranks]
    group_mean_ranks = [np.mean(r) for r in group_ranks]

    # Tie correction factor (sum over distinct ranks)
    # See: Dunn (1964); many implementations use tie correction via rank frequencies.
    # We'll compute it from the pooled ranks.
    # Frequency of each unique rank
    _, counts = np.unique(ranks, return_counts=True)
    tie_term = np.sum(counts**3 - counts)
    tie_correction = 1.0 - tie_term / (n**3 - n) if (n**3 - n) > 0 else 1.0

    # Pairwise Z statistics
    results = []
    for i in range(k):
        for j in range(i + 1, k):
            Ri, Rj = group_mean_ranks[i], group_mean_ranks[j]
            ni, nj = group_ns[i], group_ns[j]
            # Dunn's standard error with tie correction
            se = np.sqrt((n * (n + 1) / 12.0) * (1.0 / ni + 1.0 / nj) * tie_correction)
            z = (Ri - Rj) / se
            p_raw = 2.0 * (1.0 - norm.cdf(abs(z)))  # two-sided

            results.append({
                'group_i': levels[i],
                'group_j': levels[j],
                'z': z,
                'p_raw': p_raw
            })

    dunn_df = pd.DataFrame(results)

    # Multiple testing adjustment
    if not dunn_df.empty:
        m = len(dunn_df)
        if p_adjust == 'bonferroni':
            dunn_df['p_adj'] = np.minimum(dunn_df['p_raw'] * m, 1.0)
        elif p_adjust == 'holm':
            # Holm–Bonferroni
            order = np.argsort(dunn_df['p_raw'].values)
            p_sorted = dunn_df['p_raw'].values[order]
            adj = np.zeros_like(p_sorted)
            # Holm step-up
            for idx, p in enumerate(p_sorted):
                adj[idx] = min((m - idx) * p, 1.0)
            # Ensure monotonicity
            for idx in range(m - 2, -1, -1):
                adj[idx] = max(adj[idx], adj[idx + 1])
            # Put back in original order
            p_adj = np.empty_like(adj)
            p_adj[order] = adj
            dunn_df['p_adj'] = p_adj
        else:
            dunn_df['p_adj'] = dunn_df['p_raw']
    else:
        dunn_df['p_adj'] = []

    return {
        'kruskal': {'H': H, 'p': p_kw, 'epsilon2': epsilon2, 'k': k, 'n': n},
        'dunn': dunn_df
    }


def kruskal_test(combinations, df, p_adjust='holm', print_table=True):
    """
    Wrapper for multiple (continuous, categorical) pairs.
    Prints Kruskal–Wallis with epsilon^2 and Dunn's post-hoc table.

    combinations: list of (continuous_col, categorical_col)
    """
    for continuous_col, categorical_col in combinations:
        res = kruskal_effect_and_dunn(df, continuous_col, categorical_col, p_adjust=p_adjust)
        kw = res['kruskal']
        print(f"\n=== {continuous_col} ~ {categorical_col} ===")
        print(f"Kruskal–Wallis: H={kw['H']:.4f}, p={kw['p']:.4g}, epsilon^2={kw['epsilon2']:.4f} (k={kw['k']}, n={kw['n']})")

        if print_table and not res['dunn'].empty:
            # Sort by adjusted p-value for readability
            dunn_tbl = res['dunn'].sort_values('p_adj').reset_index(drop=True)
            print("Dunn post-hoc (two-sided):")
            print(dunn_tbl.to_string(index=False,
                                     formatters={'z': lambda x: f"{x:.3f}",
                                                 'p_raw': lambda x: f"{x:.4g}"}))



def perform_error_analysis(results_file: str):
    if "CorrI" in results_file:
        results_file_2 = "Benchmarking_files/Results/results_CorrI_RC.xlsx"
        print("McNemar tests...")
        mcnemar_test(results_file, results_file_2)
        print("Chi squared tests...")
        chi_squared_test_with_correction(results_file)
    if "CCR_ECI" in results_file:
        results_file_position = "Benchmarking_files/Results/results_CCR_ECI_position.xlsx"
    elif "CCR" in results_file and "ECI" not in results_file:
        results_file_position = "Benchmarking_files/Results/results_CCR_position.xlsx"
    if "CCR" in results_file:
        print("McNemar tests...")
        mcnemar_test(results_file, results_file_position)
        print("per-class performance analysis...")
        results_df = per_class_test(results_file)
        summary_df = summarize_per_class(results_df)
        print(summary_df.to_string(index=False))
        association_with_complexity_metrics(results_file)
