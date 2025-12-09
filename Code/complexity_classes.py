import pandas as pd
import networkx as nx
import math
import spacy
from scipy.stats import pearsonr
from dataset_statistics import readability_metrics
import itertools

class SemanticComplexity:
    def __init__(self, file_name: str):
        """
        Initializes the SemanticComplexity class
        """
        self.file_name = file_name
        self.df = self._read_dataset()
        self.causal_relations = self._extract_causal_relations()
        self.complexity_results_filename = 'complexity_results.csv'

    def _read_dataset(self):
        return pd.read_excel(self.file_name)

    def _extract_causal_relations(self):
        statements_and_causal_relations = list()
        statements = self.df['ipccText'].unique().tolist()
        for statement in statements:
            causal_relations = self.df[self.df['ipccText'] == statement].to_dict(orient='records')
            statements_and_causal_relations.append((statement, causal_relations))
        return statements_and_causal_relations

    # ---------- Graph Utilities ----------
    def build_graph(self, list_of_relations: list):
        G = nx.DiGraph()
        for cause, effect in list_of_relations:
            G.add_edge(cause, effect)
        return G

    def build_labeled_digraph(self, triples):
        G = nx.DiGraph()
        for cause, effect, label in triples:
            G.add_edge(cause, effect, label=label)
        return G

    def build_subgraphs(self, graph: nx.DiGraph):
        components = list(nx.weakly_connected_components(graph))
        return [graph.subgraph(c).copy() for c in components]

    def find_nodes_with_mixed_incoming_labels(self, G):
        mixed_label_nodes = []
        for node in G.nodes():
            incoming_labels = {
                G.edges[pred, node].get('label')
                for pred in G.predecessors(node)
                if G.edges[pred, node].get('label')
            }
            if len(incoming_labels) > 1:
                mixed_label_nodes.append(node)
        return mixed_label_nodes

    def find_nodes_with_mixed_outgoing_labels(self, G):
        mixed_label_nodes = []
        for node in G.nodes():
            outgoing_labels = {
                G.edges[node, succ].get('label')
                for succ in G.successors(node)
                if G.edges[node, succ].get('label')
            }
            if len(outgoing_labels) > 1:
                mixed_label_nodes.append(node)
        return mixed_label_nodes

    def find_all_consecutive_negative_paths(self, G):
        all_paths = []

        def dfs(node, path):
            extended = False
            for succ in G.successors(node):
                label = G.edges[node, succ].get('label')
                if "Negative" in label:
                    dfs(succ, path + [succ])
                    extended = True
            if not extended and len(path) > 2:
                all_paths.append(path)

        for node in G.nodes():
            dfs(node, [node])

        return all_paths

    # ---------- Complexity Calculations ----------
    def complexity_of_common_structures(self, common_list):
        graph = self.build_graph(common_list)
        subgraphs = self.build_subgraphs(graph)
        return sum(len(sg.edges()) for sg in subgraphs) + len(subgraphs)

    def complexity_of_elaboration_structures(self, elaboration_list):
        graph = self.build_graph(elaboration_list)
        subgraphs = self.build_subgraphs(graph)
        return sum(len(sg.edges()) for sg in subgraphs) + len(subgraphs)

    def complexity_of_nested_causality(self, nested_list):
        nesting_terms = set(rel[-1] for rel in nested_list)
        nested_complexity = 0
        for term in nesting_terms:
            nested_rels = [(rel[0], rel[1]) for rel in nested_list if rel[-1] == term]
            count = len(nested_rels)
            if count > 0:
                nested_complexity += count + count * math.log(count)
        return nested_complexity

    def complexity_of_polarity(self, pol_list):
        if "Negative" not in [rel[-1] for rel in pol_list]:
            return 0
        graph = self.build_labeled_digraph(pol_list)
        return (
            len([rel for rel in pol_list if "Negative" in rel[-1]]) +
            len(self.find_nodes_with_mixed_incoming_labels(graph)) +
            len(self.find_nodes_with_mixed_outgoing_labels(graph)) +
            len(self.find_all_consecutive_negative_paths(graph))
        )

    def complexity_of_correlation(self, cor_list):
        if "Negative_correlation" not in [rel[-1] for rel in cor_list]:
            return 0
        graph = self.build_labeled_digraph(cor_list)
        return (
            len([rel for rel in cor_list if "Negative_correlation" in rel[-1]]) +
            len(self.find_nodes_with_mixed_incoming_labels(graph)) +
            len(self.find_nodes_with_mixed_outgoing_labels(graph)) +
            len(self.find_all_consecutive_negative_paths(graph))
        )

    # ---------- Run Semantic Complexity ----------
    def run(self, verbose: bool = False):
        """
        Computes semantic complexity for each statement and returns structured results.
        """
        results = []
        nlp = spacy.load('en_core_web_sm')

        for statement, cr in self.causal_relations:
            cc, ex, nested, pol, cor = self.extract_semantic_features(cr)

            complexity_cc = self.complexity_of_common_structures(cc)
            complexity_ex = self.complexity_of_elaboration_structures(ex)
            complexity_nested = self.complexity_of_nested_causality(nested)
            complexity_correlation = self.complexity_of_correlation(cor)
            complexity_polarity = self.complexity_of_polarity(pol)

            total_complexity = sum([
                complexity_cc,
                complexity_ex,
                complexity_nested,
                complexity_correlation,
                complexity_polarity
            ])

            doc = nlp(statement)
            complexity_density = total_complexity / sum(1 for token in doc if token.is_alpha)

            if verbose:
                print(f"Semantic Complexity for: {statement}")
                print(f"--- CC: {complexity_cc}")
                print(f"--- EX: {complexity_ex}")
                print(f"--- Nested: {complexity_nested}")
                print(f"--- Correlation: {complexity_correlation}")
                print(f"--- Polarity: {complexity_polarity}")
                print(f"--- Total: {total_complexity}")
                print(f"--- Density: {complexity_density}\n")

            readability_scores = readability_metrics(statement)

            results.append({
                'statement': statement,
                'complexity_cc': complexity_cc,
                'complexity_ex': complexity_ex,
                'complexity_nested': complexity_nested,
                'complexity_correlation': complexity_correlation,
                'complexity_polarity': complexity_polarity,
                'total_complexity': total_complexity,
                'complexity_density': complexity_density,
                'statement_length': len(doc),
                'fre': readability_scores["Flesch Reading Ease"],
                'fkg': readability_scores["Flesch Kincaid Grade"],
                'cli': readability_scores["Coleman Liau Index"],
                'ari': readability_scores["Automated Readability Index"],
                'dcrs': readability_scores["Dale Chall Score"],
            })

        self.save_to_csv(results)

        return results

    def save_to_csv(self, l):
        df = pd.DataFrame(l)
        df.to_csv(self.complexity_results_filename, index=False)

    def extract_semantic_features(self, relations):
        """
        Extracts semantic features from the causal relations.
        Returns a tuple: (common, elaboration, nested, polarity, correlation)
        """
        common = []
        elaboration = []
        nested_relations = []
        polarity = []
        correlation = []

        for rel in relations:
            combined = rel.get('Combined')
            cause_belongs = rel.get('cause belongs to')
            effect_belongs = rel.get('effect belongs to')
            cause = rel.get('Cause_no_quantifier')
            effect = rel.get('Effect_no_quantifier')
            correlation_label = rel.get('Correlation (positive/negative or increase-increase/increase-decrease)')
            polarity_label = rel.get("Causation: Positive/Negative (causes/doesn't cause or produce/do_not_produce)")

            if pd.notna(combined):
                if combined == "Yes":
                    if pd.notna(cause_belongs):
                        common.append((cause, cause_belongs))
                    if pd.notna(effect_belongs):
                        common.append((effect, effect_belongs))
                else:
                    if rel.get('Nested causality') != "Yes":
                        if pd.notna(cause_belongs):
                            elaboration.append((cause, cause_belongs))
                        if pd.notna(effect_belongs):
                            elaboration.append((effect, effect_belongs))
                    else:
                        nested_relations.append((cause, effect, effect_belongs))

            if pd.notna(cause) and pd.notna(effect):
                if pd.notna(correlation_label):
                    correlation.append((cause, effect, correlation_label))
                if pd.notna(polarity_label):
                    polarity.append((cause, effect, polarity_label))

        return (
            list(set(common)),
            list(set(elaboration)),
            list(set(nested_relations)),
            list(set(polarity)),
            list(set(correlation))
        )

    def complexity_evaluation(self):
        df = pd.read_csv(self.complexity_results_filename)
        df_complexity = df[df['total_complexity'] > 0.0]

        print(f"Number of graphs with complexity == 0: {len(df) - len(df_complexity)}")
        print(f"Number of graphs with complexity > 0: {len(df_complexity)} ({len(df_complexity) / len(df) * 100}%)")
        print(
            f"--- C_tot > 0: {len(df[df['total_complexity'] > 0.0])}. Min: {min(df[df['total_complexity'] > 0.0]['total_complexity'])}. Max: {max(df[df['total_complexity'] > 0.0]['total_complexity'])}")
        print(f"--- C_com > 0: {len(df[df['complexity_cc'] > 0.0])}. Min: {min(df[df['complexity_cc'] > 0.0]['complexity_cc'])}. Max: {max(df[df['complexity_cc'] > 0.0]['complexity_cc'])}")
        print(f"--- C_ex > 0: {len(df[df['complexity_ex'] > 0.0])}. Min: {min(df[df['complexity_ex'] > 0.0]['complexity_ex'])}. Max: {max(df[df['complexity_ex'] > 0.0]['complexity_ex'])}")
        print(f"--- C_nest > 0: {len(df[df['complexity_nested'] > 0.0])}. Min: {min(df[df['complexity_nested'] > 0.0]['complexity_nested'])}. Max: {max(df[df['complexity_nested'] > 0.0]['complexity_nested'])}")
        print(f"--- C_corr > 0: {len(df[df['complexity_correlation'] > 0.0])}. Min: {min(df[df['complexity_correlation'] > 0.0]['complexity_correlation'])}. Max: {max(df[df['complexity_correlation'] > 0.0]['complexity_correlation'])}")
        print(f"--- C_pol > 0: {len(df[df['complexity_polarity'] > 0.0])}. Min: {min(df[df['complexity_polarity'] > 0.0]['complexity_polarity'])}. Max: {max(df[df['complexity_polarity'] > 0.0]['complexity_polarity'])}")

        self.analyze_correlation(df, 'statement_length', 'total_complexity')
        self.analyze_correlation(df, 'total_complexity', 'fre')
        self.analyze_correlation(df, 'total_complexity', 'fkg')
        self.analyze_correlation(df, 'total_complexity', 'cli')
        self.analyze_correlation(df, 'total_complexity', 'ari')
        self.analyze_correlation(df, 'total_complexity', 'dcrs')

        df_complexity['non_zero_count'] = (df_complexity[["complexity_cc", "complexity_ex", "complexity_nested", "complexity_correlation", "complexity_polarity"]] != 0.0).sum(axis=1)
        counter_contributing_metrics = self.count_occurrences(df_complexity, "non_zero_count")
        print(counter_contributing_metrics)
        self.analyze_complexity_combinations(df_complexity)

    def count_occurrences(self, df, column, values=[1,2,3,4,5]):
        counts = dict()
        for value in values:
            counts[value] = int((df[column] == value).sum())
        return counts

    def analyze_correlation(self, df, att1, att2):
        corr, p_value = pearsonr(df[att1], df[att2])
        print(
            f"Correlation between {att1} and {att2}. Pearson correlation: {corr:.3f}, p-value: {p_value:.4f}")

    def analyze_complexity_combinations(self, df):
        #filtered = df[df["non_zero_count"] >= 2]
        cols = ["complexity_cc", "complexity_ex", "complexity_nested", "complexity_correlation", "complexity_polarity"]
        co_occurrence = pd.DataFrame(0, index=cols, columns=cols)

        # For each row, print only the columns with value > 0
        for idx, row in df.iterrows():
            active = [col for col in cols if row[col] > 0]
            for a, b in itertools.combinations(active, 2):
                co_occurrence.loc[a, b] += 1
                co_occurrence.loc[b, a] += 1
            for a in active:  # diagonal counts (self-occurrence)
                co_occurrence.loc[a, a] += 1
        co_occurrence.to_csv("co-occurrence_matrix_complexity_metrics.csv")

def main():
    rerun_metrics = False
    semantic = SemanticComplexity(file_name="Full_IPCC_DATASET.xlsx")
    if rerun_metrics:
        semantic.run(verbose=True)
    else:
        semantic.complexity_evaluation()

main()