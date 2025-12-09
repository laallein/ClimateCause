import pandas as pd
from collections import defaultdict
import numpy as np
import textstat

def extract_unique_cause_effect_pairs(df):
    print("Number of cause-effect pairs (with duplicates): ", len(list(df[['Cause--NP', 'Effect--NP']].itertuples(index=False, name=None))))
    # Drop duplicate pairs
    unique_pairs = df[['Cause--NP', 'Effect--NP']].drop_duplicates()
    pairs = list(unique_pairs.itertuples(index=False, name=None))
    print("Number of unique cause-effect pairs as mentioned in text: ", len(pairs))

def extract_unique_cause_effect_pairs_normalized(df):
    unique_pairs = df[['Cause_no_modifier', 'Effect_no_modifier']].drop_duplicates()
    pairs = list(unique_pairs.itertuples(index=False, name=None))
    print("Number of unique cause-effect pairs, normalized: ", len(pairs))
    unique_pairs_with_correlation = df[['Cause_no_modifier', 'Effect_no_modifier', 'Correlation (positive/negative or increase-increase/increase-decrease)']].drop_duplicates()
    pairs_norm = list(unique_pairs_with_correlation.itertuples(index=False, name=None))
    print("(if same as above, no conflicts in correlation) Number of unique cause-effect pairs, normalized: ", len(pairs_norm))
    for (c, e, cor) in pairs_norm:
        if cor == 'Positive_correlation':
            if (c, e, 'Negative_correlation') in pairs_norm:
                print((c, e, cor))
    positive_correlations = unique_pairs_with_correlation[unique_pairs_with_correlation['Correlation (positive/negative or increase-increase/increase-decrease)'] == 'Positive_correlation'].values.tolist()
    print("--- Positive correlations (unique): ", len(positive_correlations))
    negative_correlations = unique_pairs_with_correlation[unique_pairs_with_correlation[
                                                              'Correlation (positive/negative or increase-increase/increase-decrease)'] == 'Negative_correlation'].values.tolist()
    print("--- Negative correlations (unique): ", len(negative_correlations))
    positive_correlations = df[df['Correlation (positive/negative or increase-increase/increase-decrease)'] == 'Positive_correlation'].values.tolist()
    print("--- Positive correlations (all): ", len(positive_correlations))
    negative_correlations = df[df['Correlation (positive/negative or increase-increase/increase-decrease)'] == 'Negative_correlation'].values.tolist()
    print("--- Negative correlations (all): ", len(negative_correlations))
    ### Analyze polarity of causal relations
    unique_pairs_pol = df[['Cause--NP', 'Effect--NP', "Causation: Positive/Negative (causes/doesn't cause or produce/do_not_produce)"]] #.drop_duplicates()
    positive_pol = unique_pairs_pol[unique_pairs_pol["Causation: Positive/Negative (causes/doesn't cause or produce/do_not_produce)"] == 'Positive'].values.tolist()
    print("--- Positive polarity: ", len(positive_pol))
    negative_pol = unique_pairs_pol[unique_pairs_pol["Causation: Positive/Negative (causes/doesn't cause or produce/do_not_produce)"] == 'Negative'].values.tolist()
    print("--- Negative polarity: ", len(negative_pol))
    ### Count nested causality
    unique_pairs_nested = df[df["Nested causality"] == 'Yes'].drop_duplicates().values.tolist()
    print("--- Nested causality (unique): ", len(unique_pairs_nested))
    pairs_nested = df[df["Nested causality"] == 'Yes'].values.tolist()
    print("--- Nested causality (all): ", len(pairs_nested))
    ### Count implicit causality
    explicit_pairs = df[df["Explicit/implicit"] == 'E'].values.tolist()
    implicit_pairs = df[df["Explicit/implicit"] == 'I'].values.tolist()
    print("--- Explicit causal relations (all): ", len(explicit_pairs))
    print("--- Implicit causal relations (all): ", len(implicit_pairs))
    ### Count unique target words with their counts
    unique_targets = df["Trigger"].unique().tolist()
    print("--- Number of unique target words: ", len(unique_targets))

    subordinating = df[(df['cause belongs to'].notna() & (df['cause belongs to'] != '')) |
       (df['effect belongs to'].notna() & (df['effect belongs to'] != ''))]
    print("--- Number of subordinating targets: ", subordinating.shape[0])
    elaboration_of_examples = subordinating[subordinating['Combined'] == 'No'].values.tolist()
    common = subordinating[subordinating['Combined'] == 'Yes'].values.tolist()
    print("--- Elaboration of examples: ", len(elaboration_of_examples))
    print("--- Common: ", len(common))


def extract_unique_paragraphs(df):
    paragraph_ids = df['Paragraph'].unique().tolist()
    print("Number of paragraphs:", len(paragraph_ids))

def extract_unique_statements(df):
    statements = df[['Paragraph', 'Series ordinal', 'Causal?']].drop_duplicates()
    print("Number of statements: ", len(list(statements.itertuples(index=False, name=None))))
    with_c = statements[statements['Causal?'] == 'Yes'].values.tolist()
    print("--- with causality:", len(with_c))
    without_c = statements[statements['Causal?'] == 'No'].values.tolist()
    print("--- without causality:", len(without_c))
    return df['ipccText'].drop_duplicates().tolist()

def extract_unique_sections(df):
    l = df['Part of'].unique().tolist()
    print("Number of Sections:", len(l))
    print("Section titles: ", l)

def analyze_complex_causal_structures(df):
    chain_analysis(df)

def chain_analysis(df):
    chain_df = df.groupby('ipccText')['Chain'].value_counts().unstack(fill_value=0)
    paragraphs_with_yes = chain_df[chain_df["Yes"] > 0].index.tolist()
    print("Number of paragraphs with causal chains: ", len(paragraphs_with_yes))
    all_chains = list()
    all_correlation_sequences = list()
    correlation_overall = list()
    for paragraph in paragraphs_with_yes:
        d = df[df['ipccText'] == paragraph]
        cr = list(d[['Cause_no_modifier', 'Effect_no_modifier']].itertuples(index=False, name=None))
        chains = sample_chains(cr)
        correlation_sequences_in_chains = correlation_in_chains(chains, d)
        all_chains.extend(chains)
        all_correlation_sequences.extend(correlation_sequences_in_chains)
        for seq in correlation_sequences_in_chains:
            if 'Positive_correlation' in seq and 'Negative_correlation' not in seq:
                correlation_overall.append('Only_positive_correlation')
            elif 'Positive_correlation' in seq and 'Negative_correlation' in seq:
                correlation_overall.append('Mixed_correlation')
            else:
                correlation_overall.append('Only_negative_correlation')

    print("Total number of chains: ", len(all_chains))
    print("--- event count = 3: ", len([chain for chain in all_chains if len(chain) == 3]))
    print("--- event count > 3: ", len([chain for chain in all_chains if len(chain) > 3]))
    print("Correlation sequences in chains")
    print("--- All positive: ", len([c for c in correlation_overall if c == 'Only_positive_correlation']))
    print("--- All negative: ", len([c for c in correlation_overall if c == 'Only_negative_correlation']))
    print("--- Mixed: ", len([c for c in correlation_overall if c == 'Mixed_correlation']))

def correlation_in_chains(chains, df):
    all_correlation_sequences = list()
    for chain in chains:
        correlation_sequence = list()
        for i in range(len(chain[:-1])):
            filtered_df = df[(df['Cause_no_modifier'] == chain[i]) & (df['Effect_no_modifier'] == chain[i+1])]
            correlation = filtered_df['Correlation (positive/negative or increase-increase/increase-decrease)'].values.tolist()[0]
            correlation_sequence.append(correlation)
        all_correlation_sequences.append(correlation_sequence)
    return all_correlation_sequences

def sample_chains(pairs: list):
    # Normalize text to lowercase for consistent matching
    normalized_pairs = [(cause.lower(), effect.lower()) for cause, effect in pairs]

    # Build a mapping from cause to effect
    cause_to_effect = defaultdict(list)
    for cause, effect in normalized_pairs:
        cause_to_effect[cause].append(effect)

    # Function to find causal chains recursively
    def find_chains(start, visited=None):
        if visited is None:
            visited = set()
        visited.add(start)
        chains = []
        for effect in cause_to_effect.get(start, []):
            if effect not in visited:
                subchains = find_chains(effect, visited.copy())
                if subchains:
                    for chain in subchains:
                        chains.append([start] + chain)
                else:
                    chains.append([start, effect])
        return chains

    # Identify root causes (causes that are not effects)
    all_effects = set(effect for _, effect in normalized_pairs)
    root_causes = set(cause for cause, _ in normalized_pairs if cause not in all_effects)

    # Find all causal chains starting from root causes
    causal_chains = []
    for root in root_causes:
        causal_chains.extend(find_chains(root))

    only_chains = [chain for chain in causal_chains if len(chain) > 2]
    return only_chains

def readability_metrics(text: str):
    return {"Flesch Reading Ease": textstat.flesch_reading_ease(text),
            "Flesch Kincaid Grade": textstat.flesch_kincaid_grade(text),
            "Gunning Fog Index": textstat.gunning_fog(text), "SMOG Index": textstat.smog_index(text),
            "Coleman Liau Index": textstat.coleman_liau_index(text),
            "Automated Readability Index": textstat.automated_readability_index(text),
            "Dale Chall Score": textstat.dale_chall_readability_score(text),
            "SMOG Index": textstat.smog_index(text),
            "Reading time": textstat.reading_time(text, ms_per_char=14.69)
            }


def compute_readability(statements: list):
    flesch_reading, flesch_kincaid, gunning, coleman, ari, dale, smog, rt = list(),list(), list(), list(), list(), list(), list(), list()
    for s in statements:
        all_metrics = readability_metrics(s)
        flesch_reading.append(all_metrics['Flesch Reading Ease'])
        flesch_kincaid.append(all_metrics['Flesch Kincaid Grade'])
        gunning.append(all_metrics['Gunning Fog Index'])
        coleman.append(all_metrics['Coleman Liau Index'])
        ari.append(all_metrics['Automated Readability Index'])
        dale.append(all_metrics['Dale Chall Score'])
        smog.append(all_metrics['SMOG Index'])
        rt.append(all_metrics['Reading time'])

    print("READABILITY METRICS")
    print("Flesch Reading Ease: ", readability_stats(flesch_reading))
    print("Flesch Kincaid Grade: ", readability_stats(flesch_kincaid))
    #print("Gunning Fog Index: ", readability_stats(gunning))
    print("Coleman Liau Index: ", readability_stats(coleman))
    print("Automated Readability Index: ", readability_stats(ari))
    print("Dale Chall Score: ", readability_stats(dale))
    #print("SMOG Index: ", readability_stats(smog))
    #print("Reading time: ", readability_stats(rt))


def readability_stats(results: list):
    return [float(np.median(results)), float(np.mean(results)), float(np.std(results)), min(results), max(results)]


def main():
    paragraph_df = pd.read_excel('Full_IPCC_DATASET.xlsx')
    extract_unique_sections(paragraph_df)
    extract_unique_paragraphs(paragraph_df)
    statement_texts = extract_unique_statements(paragraph_df)
    compute_readability(statement_texts)
    extract_unique_cause_effect_pairs(paragraph_df)
    extract_unique_cause_effect_pairs_normalized(paragraph_df)
    print("------")


if __name__ == '__main__':
    main()