import json
import re
import pandas as pd
import os

def load_jsonl(file_path):
    with open(file_path, 'r') as f:
        return [json.loads(line.strip()) for line in f]

def extract_label(text, target, potential_labels):
    if target == "would":
        pattern = rf'\b{re.escape(target)}\b\s+(\w+)'
        match = re.search(pattern, text)
        if match:
            answer = match.group(1)
            if answer in potential_labels:
                return answer

    pattern = rf'\b(\w+)\s+{re.escape(target)}\b'
    match = re.search(pattern, text)
    if match:
        answer = match.group(1)
        if answer in potential_labels:
            return answer
        matches = re.findall(pattern, text)
        for match in matches:
            if match in potential_labels:
                return match

    # directions
    if target == "direction":
        target = "directions"
        pattern = rf'\b(\w+)\s+{re.escape(target)}\b'
        match = re.search(pattern, text)
        if match:
            answer = match.group(1)
            if answer in potential_labels:
                return answer

    label_1 = rf'\b(\w+)\s+{re.escape(potential_labels[0])}\b'
    label_2 = rf'\b(\w+)\s+{re.escape(potential_labels[1])}\b'
    if re.search(label_1, text, re.IGNORECASE) and not re.search(label_2, text, re.IGNORECASE):
        return potential_labels[0]
    elif re.search(label_2, text, re.IGNORECASE) and not re.search(label_1, text, re.IGNORECASE):
        return potential_labels[1]
    elif text == potential_labels[0]:
        return potential_labels[0]
    elif text == potential_labels[1]:
        return potential_labels[1]
    else:
        target = "MASK"
        pattern = rf'\b{re.escape(target)}\b\s+(\w+)'
        matches = re.findall(pattern, text)
        for match in matches:
            if match in potential_labels:
                return match

        target = "Answer"
        pattern = rf'\b{re.escape(target)}\b\s+(\w+)'
        match = re.search(pattern, text)
        if match:
            answer = match.group(1)
            if answer in potential_labels:
                return answer

        pattern = r"\b(" + "|".join(re.escape(c) for c in potential_labels) + r")\b"
        # Find all matches; take the last one if any
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        if matches:
            return matches[-1].lower()

        if "nswernone" in text or "nswerNone" in text:
            return "none"
        if "nswerstart" in text or "nswerStart" in text:
            return "start"
        if "nswerend" in text or "nswerEnd" in text:
            return "end"
        if "nswermiddle" in text or "nswerMiddle" in text:
            return "middle"
        if "nswerYes" in text or "nsweryes" in text:
            return "Yes"
        if "nswerNo" in text or "nswerno" in text:
            return "No"

        return None

def normalize_results(batch_output, label_options, target):
    results = {
        'custom_id' : list(),
        'label': list()
    }
    for r in batch_output:
        results['custom_id'].append(r['custom_id'])
        response = re.sub(r'[^a-zA-Z0-9\s]', '', r['response']['body']['choices'][0]['message']['content'])
        label = extract_label(response, target, label_options)
        print(response)
        print(f'LABEL: {label}')
        if label is None:
            results['label'].append(None)
        else:
            results['label'].append(label_options.index(label.lower()))

    return results

def create_results_file(dataset_file, results_file):
    master_df = pd.read_excel(dataset_file)
    master_df = master_df[master_df['Causal?'] == 'Yes']

    if "CorrI" in results_file:
        master_df['Correlation (positive/negative or increase-increase/increase-decrease)'] = (
            master_df['Correlation (positive/negative or increase-increase/increase-decrease)'].replace({
            re.compile(r'^positive_correlation$', re.IGNORECASE): 1,
            re.compile(r'^negative_correlation$', re.IGNORECASE): 0
        }, regex=True))

    if "member" in results_file:
        master_df['chain_membership_label'] = (
            master_df['chain_membership_label'].replace({
                re.compile(r'^yes$', re.IGNORECASE): 1,
                re.compile(r'^no$', re.IGNORECASE): 0
            }, regex=True))

    if "position" in results_file:
        master_df['chain_position_label'] = (
            master_df['chain_position_label'].replace({
                re.compile(r'^none$', re.IGNORECASE): 0,
                re.compile(r'^start$', re.IGNORECASE): 1,
                re.compile(r'^middle$', re.IGNORECASE): 2,
                re.compile(r'^end$', re.IGNORECASE): 3
            }, regex=True))

    master_df.to_excel(results_file, index=False)

def merge_results(res, output_file, column_name):
    big_df = pd.read_excel(output_file)
    res_df = pd.DataFrame(res)

    df = big_df.merge(res_df, on='custom_id', how='right')
    df[column_name] = df['label']
    df = df.drop(columns=['label'])
    df.to_excel(output_file, index=False)

    return df


def compute_metrics_and_save(df, predicted_labels_col, prompting_strategy, output_file):
    y_pred = df[predicted_labels_col]
    if "CorrI" in prompting_strategy:
        y_true = df['Correlation (positive/negative or increase-increase/increase-decrease)']
    elif "member" in prompting_strategy:
        y_true = df['chain_membership_label']
    else:
        y_true = df['chain_position_label']


    # Calculate TP, FP, FN
    TP = ((y_pred == 1) & (y_true == 1)).sum()
    FP = ((y_pred == 1) & (y_true == 0)).sum()
    FN = ((y_pred == 0) & (y_true == 1)).sum()

    # Compute precision, recall, F1
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    F1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # Create results DataFrame
    results_df = pd.DataFrame({
        'prompting_strategy': [prompting_strategy],
        'precision': [precision],
        'recall': [recall],
        'F1': [F1]
    })

    print(results_df)
    # Save to CSV
    if os.path.exists(output_file):
        master_df = pd.read_csv(output_file)
        results_df = pd.concat([master_df, results_df], ignore_index=True)

    results_df.to_csv(output_file, index=False)

def obtain_metrics(sub_df: pd.DataFrame,
                   results: dict,
                   strategy: str):
    mean_precision = sub_df['precision'].mean()
    std_precision = sub_df['precision'].std()

    mean_recall = sub_df['recall'].mean()
    std_recall = sub_df['recall'].std()

    mean_f1 = sub_df['F1'].mean()
    std_f1 = sub_df['F1'].std()

    results['prompting_strategy'].append(strategy)
    results['precision'].append(f"{mean_precision:.4f} ± {std_precision:.4f}")
    results['recall'].append(f"{mean_recall:.4f} ± {std_recall:.4f}")
    results['F1'].append(f"{mean_f1:.4f} ± {std_f1:.4f}")

    return results

def compile_results(f: str):
    df = pd.read_csv(f)
    results = {'prompting_strategy': list(),
               'precision': list(),
               'recall': list(),
               'F1': list()}

    for strategy in ["CorrI_0", "CorrI_F", "CorrI_CoT", "CorrI_RC_0", "CorrI_RC_F", "CorrI_RC_CoT",
                     "CCR_member_A", "CCR_member_SN", "CCR_member_ML", "CCR_position_A", "CCR_position_SN", "CCR_position_ML",
                     "CCR_ECI_member_0", "CCR_ECI_member_F", "CCR_ECI_member_CoT",
                     "CCR_ECI_position_0", "CCR_ECI_position_F", "CCR_ECI_position_CoT",]:
        sub_df = df[df['prompting_strategy'].str.contains(strategy, case=False, na=False)]
        results = obtain_metrics(sub_df, results, strategy=strategy)

    corrI_df = df[df['prompting_strategy'].str.contains("CorrI", case=False, na=False)]
    RC_df = corrI_df[corrI_df['prompting_strategy'].str.contains("RC", case=False, na=False)]
    results = obtain_metrics(RC_df, results, strategy="CorrI_RC_average")
    no_RC_df = corrI_df[~corrI_df['prompting_strategy'].str.contains("RC", case=False, na=False)]
    results = obtain_metrics(no_RC_df, results, strategy="CorrI_average")
    CCR_member_df = df[df['prompting_strategy'].str.contains("CCR_member", case=False, na=False)]
    results = obtain_metrics(CCR_member_df, results, strategy="CCR_member_average")
    CCR_position_df = df[df['prompting_strategy'].str.contains("CCR_position", case=False, na=False)]
    results = obtain_metrics(CCR_position_df, results, strategy="CCR_position_average")
    CCR_ECI_member_df = df[df['prompting_strategy'].str.contains("ECI_member", case=False, na=False)]
    results = obtain_metrics(CCR_ECI_member_df, results, strategy="CCR_ECI_member_average")
    CCR_ECI_position_df = df[df['prompting_strategy'].str.contains("ECI_position", case=False, na=False)]
    results = obtain_metrics(CCR_ECI_position_df, results, strategy="CCR_ECI_position_average")

    results_df = pd.DataFrame(results)
    print(results_df)

def postprocessing(task: str,
                   prompting_strategy: str,
                   batch_file: str,
                   dataset_file: str,
                   results_file: str,
                   results_masterfile: str,
                   labels: list,
                   target: str):
    print("Normalize results...")
    batch_output = load_jsonl(batch_file)
    results = normalize_results(batch_output, labels, target)
    print("Add results to results file...")
    if not os.path.exists(results_file):
        create_results_file(task + "_" + dataset_file, results_file)
    label_column = task + prompting_strategy + "_labels"
    results_df = merge_results(results, results_file, column_name=label_column)
    print("Perform evaluation...")
    compute_metrics_and_save(results_df, label_column, task + prompting_strategy, results_masterfile)