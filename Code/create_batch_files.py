import pandas as pd
import json
import os
import networkx as nx
import io

def generate_prompt(task, e_i, e_j, s):
    prompts = {}

    if "CorrI" in task:
        # CorrI variants
        prompts.update({
            "CorrI_0_1": f"[MASK] in {{positive, negative}}. There is a [MASK] correlation between {e_i} and {e_j}.",
            "CorrI_0_2": f"[MASK] in {{same, opposite}}. {e_i} impact(s) {e_j}. When we would intervene in {e_i}, {e_i} and {e_j} change in the [MASK] direction.",
            "CorrI_0_3": f"[MASK] in {{increase, decrease}}. If {e_i} (the cause) were to increase, {e_j} (the effect) would [MASK].",
            "CorrI_F_1": f"There is a positive correlation between access to finance and climate resilient development. There is a negative correlation between persistent barriers and political feasibility of deploying AFOLU mitigation options. There is a [MASK] correlation between {e_i} and {e_j}.",
            "CorrI_F_2": f"Access to finance impacts climate resilient development. When we would intervene in access to finance, access to finance and climate resilient development change in the same direction. Persistent barriers impact political feasibility of deploying AFOLU mitigation options. When we would intervene in persistent barriers, persistent barriers and political feasibility of deploying AFOLU mitigation options change in the opposite direction. {e_i} impact(s) {e_j}. When we would intervene in {e_i}, {e_i} and {e_j} change in the [MASK] direction.",
            "CorrI_F_3": f"If access to finance (the cause) were to increase, climate resilient development (the effect) would increase. If persistent barriers (the cause) were to increase, political feasibility of deploying AFOLU mitigation options (the effect) would decrease. If {e_i} (the cause) were to increase, {e_j} (the effect) would [MASK].",
            "CorrI_CoT_1": f"Let's think step by step. [MASK] in {{positive, negative}}. There is a [MASK] correlation between {e_i} and {e_j}.",
            "CorrI_CoT_2": f"Let's think step by step. [MASK] in {{same, opposite}}. {e_i} impact(s) {e_j}. When we would intervene in {e_i}, {e_i} and {e_j} change in the [MASK] direction.",
            "CorrI_CoT_3": f"Let's think step by step. [MASK] in {{increase, decrease}}. If {e_i} (the cause) were to increase, {e_j} (the effect) would [MASK]."
        })

        # CorrI+RC variants
        prompts.update({
            "CorrI_RC_0_1": f"[MASK] in {{positive, negative}}. Statement: {s}. The statement reports a [MASK] correlation between {e_i} and {e_j}.",
            "CorrI_RC_0_2": f"[MASK] in {{same, opposite}}. Statement: {s}. {e_i} impact(s) {e_j}. Based on the statement, when we would intervene in {e_i}, {e_i} and {e_j} change in the [MASK] direction.",
            "CorrI_RC_0_3": f"[MASK] in {{increase, decrease}}. Given the statement: {s}. If {e_i} (the cause) were to increase, {e_j} (the effect) would [MASK].",
            "CorrI_RC_F_1": f"Statement: \"Climate resilient development is enabled by increased international cooperation including mobilising and enhancing access to finance, particularly for developing countries, vulnerable regions, sectors and groups and aligning finance flows for climate action to be consistent with ambition levels and funding needs.\". The statement reports a positive correlation between access to finance and climate resilient development. Statement: \"Persistent and region-specific barriers also continue to hamper the economic and political feasibility of deploying AFOLU mitigation options.\". The statement reports a negative correlation between persistent barriers and political feasibility of deploying AFOLU mitigation options. Statement: \"{s}\". The statement reports a [MASK] correlation between {e_i} and {e_j}.",
            "CorrI_RC_F_2": f"Statement: \"Climate resilient development is enabled by increased international cooperation including mobilising and enhancing access to finance, particularly for developing countries, vulnerable regions, sectors and groups and aligning finance flows for climate action to be consistent with ambition levels and funding needs.\". Access to finance impacts climate resilient development. Based on the statement, when we would intervene in access to finance, access to finance and climate resilient development change in the same direction. Statement: \"Persistent and region-specific barriers also continue to hamper the economic and political feasibility of deploying AFOLU mitigation options.\". Persistent barriers impact political feasibility of deploying AFOLU mitigation options. Based on the statement, when we would intervene in persistent barriers, persistent barriers and political feasibility of deploying AFOLU mitigation options change in the opposite direction. Statement: \"{s}\". {e_i} impact(s) {e_j}. Based on the statement, when we would intervene in {e_i}, {e_i} and {e_j} change in the [MASK] direction.",
            "CorrI_RC_F_3": f"Given the statement: \"Climate resilient development is enabled by increased international cooperation including mobilising and enhancing access to finance, particularly for developing countries, vulnerable regions, sectors and groups and aligning finance flows for climate action to be consistent with ambition levels and funding needs.\". If access to finance (the cause) were to increase, climate resilient development (the effect) would increase. Given the statement: \"Persistent and region-specific barriers also continue to hamper the economic and political feasibility of deploying AFOLU mitigation options.\". If persistent barriers (the cause) were to increase, political feasibility of deploying AFOLU mitigation options (the effect) would decrease. Given the statement: \"{s}\". If {e_i} (the cause) were to increase, {e_j} (the effect) would [MASK].",
            "CorrI_RC_CoT_1": f"Let's think step by step. [MASK] in {{positive, negative}}. Statement: {s}. The statement reports a [MASK] correlation between {e_i} and {e_j}.",
            "CorrI_RC_CoT_2": f"Let's think step by step. [MASK] in {{same, opposite}}. Statement: {s}. {e_i} impact(s) {e_j}. Based on the statement, when we would intervene in {e_i}, {e_i} and {e_j} change in the [MASK] direction.",
            "CorrI_RC_CoT_3": f"Let's think step by step. Statement: {s}. [MASK] in {{increase, decrease}}. If {e_i} (the cause) were to increase, {e_j} (the effect) would [MASK]."
        })

    return prompts


def generate_prompt_causal_graph(task, e_i, G_A, G_SN, G_ML, s="", event_list=""):
    prompts = {}

    if "ECI" in task:
        prompts.update({
            "CCR_ECI_member_0_4": f"Statement: \"{s}\". All causal events in the statement: {event_list}. Based on the statement and the provided list of causal events, determine whether {e_i} is reported as part of a causal chain. A causal chain is a directed path of at least three nodes in a causal graph. Give an answer within <Answer> {{Yes, No}} </Answer>.",
            "CCR_ECI_member_0_5": f"Given a statement: \"{s}\" and a list of all causal events: {event_list}. Determine whether {e_i} belongs to a causal chain reported, either explicitly or implicitly, in the statement. A causal chain is a directed path of at least three nodes in a causal graph. Answer with {{Yes, No}} only.",
            "CCR_ECI_member_0_6": f"The following list of causal events {event_list} can be found in statement \"{s}\". Does the statement report a causal chain structure that includes {e_i}? A causal chain is a directed path of at least three nodes in a causal graph. Answer with {{Yes, No}} only.",
            "CCR_ECI_member_F_4": f"A causal chain is a directed path of at least three nodes in a causal graph. Statement: \"Persistent and region-specific barriers also continue to hamper the economic and political feasibility of deploying AFOLU mitigation options.\". All causal events in the statement: [persistent barriers, region-specific barriers, economic feasibility of deploying AFOLU mitigation options, political feasibility of deploying AFOLU mitigation options]. Based on the statement and the provided list of causal events, is region-specific barriers reported as part of a causal chain? No. Statement: \"Adaptation options can become maladaptive due to their environmental impacts that constrain ecosystem services and decrease biodiversity and ecosystem resilience to climate change or by causing adverse outcomes for different groups, exacerbating inequity.\". All causal events in the statement: [adaptation options, environmental impacts, ecosystem services, biodiversity, ecosystem resilience to climate change, adverse outcomes for different groups, inequity, maladaptive adaptation options]. Based on the statement and the provided list of causal events, is inequity reported as part of a causal chain? Yes. Statement: \"{s}\". All causal events in the statement: {event_list}. Based on the statement and the provided list of causal events, is {e_i} reported as part of a causal chain?",
            "CCR_ECI_member_F_5": f"A causal chain is a directed path of at least three nodes in a causal graph. Given a statement: \"Persistent and region-specific barriers also continue to hamper the economic and political feasibility of deploying AFOLU mitigation options.\" and a list of all causal events: [persistent barriers, region-specific barriers, economic feasibility of deploying AFOLU mitigation options, political feasibility of deploying AFOLU mitigation options]. Determine whether region-specific barriers belongs to a causal chain reported, either explicitly or implicitly, in the statement. Answer: No. Given a statement: \"Adaptation options can become maladaptive due to their environmental impacts that constrain ecosystem services and decrease biodiversity and ecosystem resilience to climate change or by causing adverse outcomes for different groups, exacerbating inequity.\" and a list of all causal events: [adaptation options, environmental impacts, ecosystem services, biodiversity, ecosystem resilience to climate change, adverse outcomes for different groups, inequity, maladaptive adaptation options]. Determine whether inequity belongs to a causal chain reported, either explicitly or implicitly, in the statement. Answer: Yes.  Given a statement: \"{s}\" and a list of all causal events: {event_list}. Determine whether {e_i} belongs to a causal chain reported, either explicitly or implicitly, in the statement. Answer:",
            "CCR_ECI_member_F_6": f"A causal chain is a directed path of at least three nodes in a causal graph. The following list of causal events [persistent barriers, region-specific barriers, economic feasibility of deploying AFOLU mitigation options, political feasibility of deploying AFOLU mitigation options] can be found in statement \"Persistent and region-specific barriers also continue to hamper the economic and political feasibility of deploying AFOLU mitigation options.\". Does the statement report a causal chain structure that includes region-specific barriers? No. The following list of causal events [adaptation options, environmental impacts, ecosystem services, biodiversity, ecosystem resilience to climate change, adverse outcomes for different groups, inequity, maladaptive adaptation options] can be found in statement \"Adaptation options can become maladaptive due to their environmental impacts that constrain ecosystem services and decrease biodiversity and ecosystem resilience to climate change or by causing adverse outcomes for different groups, exacerbating inequity.\". Does the statement report a causal chain structure that includes inequity? Yes. The following list of causal events {event_list} can be found in statement \"{s}\". Does the statement report a causal chain structure that includes {e_i}?",
            "CCR_ECI_member_CoT_4": f"Statement: \"{s}\". All causal events in the statement: {event_list}. Based on the statement and the provided list of causal events, determine whether {e_i} is reported as part of a causal chain. A causal chain is a directed path of at least three nodes in a causal graph. Think step by step. Give reasoning and then give an answer within <Answer> {{Yes, No}} </Answer>.",
            "CCR_ECI_member_CoT_5": f"Given a statement: \"{s}\" and a list of all causal events: {event_list}. Determine whether {e_i} belongs to a causal chain reported, either explicitly or implicitly, in the statement. A causal chain is a directed path of at least three nodes in a causal graph. Think step by step. Finally, answer with {{Yes, No}}.",
            "CCR_ECI_member_CoT_6": f"The following list of causal events {event_list} can be found in statement \"{s}\". Does the statement report a causal chain structure that includes {e_i}? A causal chain is a directed path of at least three nodes in a causal graph. Let's think step by step and answer with {{Yes, No}}.",

        })

        prompts.update({
            "CCR_ECI_position_0_4": f"Statement: \"{s}\". All causal events in the statement: {event_list}. Based on the statement and the provided list of causal events, determine whether {e_i} is reported as part of a causal chain and, if yes, which position it holds in that chain. A causal chain is a directed path of at least three nodes in a causal graph. Give an answer within <Answer> {{start, middle, end, none}} </Answer>.",
            "CCR_ECI_position_0_5": f"Given a statement: \"{s}\" and a list of all causal events: {event_list}. Determine whether {e_i} belongs to a causal chain reported, either explicitly or implicitly, in the statement and which position it holds in that chain. A causal chain is a directed path of at least three nodes in a causal graph. Answer with {{start, middle, end, none}} only.",
            "CCR_ECI_position_0_6": f"The following list of causal events {event_list} can be found in statement \"{s}\". Does the statement report a causal chain structure that includes {e_i} and at which position in the chain can the event be found? A causal chain is a directed path of at least three nodes in a causal graph. Answer with {{start, middle, end, none}} only.",
            "CCR_ECI_position_F_4": f"A causal chain is a directed path of at least three nodes in a causal graph. Statement: \"Persistent and region-specific barriers also continue to hamper the economic and political feasibility of deploying AFOLU mitigation options.\". All causal events in the statement: [persistent barriers, region-specific barriers, economic feasibility of deploying AFOLU mitigation options, political feasibility of deploying AFOLU mitigation options]. Based on the statement and the provided list of causal events, is region-specific barriers reported as part of a causal chain, and, if yes, which position it holds in that chain? none. Statement: \"Adaptation options can become maladaptive due to their environmental impacts that constrain ecosystem services and decrease biodiversity and ecosystem resilience to climate change or by causing adverse outcomes for different groups, exacerbating inequity.\". All causal events in the statement: [adaptation options, environmental impacts, ecosystem services, biodiversity, ecosystem resilience to climate change, adverse outcomes for different groups, inequity, maladaptive adaptation options]. Based on the statement and the provided list of causal events, is inequity reported as part of a causal chain, and, if yes, which position it holds in that chain? none? middle. Statement: \"{s}\". All causal events in the statement: {event_list}. Based on the statement and the provided list of causal events, is {e_i} reported as part of a causal chain, and, if yes, which position it holds in that chain? Answer with {{start, middle, end, none}} only.",
            "CCR_ECI_position_F_5": f"A causal chain is a directed path of at least three nodes in a causal graph. Given a statement: \"Persistent and region-specific barriers also continue to hamper the economic and political feasibility of deploying AFOLU mitigation options.\" and a list of all causal events: [persistent barriers, region-specific barriers, economic feasibility of deploying AFOLU mitigation options, political feasibility of deploying AFOLU mitigation options]. Determine whether region-specific barriers belongs to a causal chain reported, either explicitly or implicitly, in the statement and answer with the position the event holds in that chain. Answer: none. Given a statement: \"Adaptation options can become maladaptive due to their environmental impacts that constrain ecosystem services and decrease biodiversity and ecosystem resilience to climate change or by causing adverse outcomes for different groups, exacerbating inequity.\" and a list of all causal events: [adaptation options, environmental impacts, ecosystem services, biodiversity, ecosystem resilience to climate change, adverse outcomes for different groups, inequity, maladaptive adaptation options]. Determine whether inequity belongs to a causal chain reported, either explicitly or implicitly, in the statement  and answer with the position the event holds in that chain. Answer: middle.  Given a statement: \"{s}\" and a list of all causal events: {event_list}. Determine whether {e_i} belongs to a causal chain reported, either explicitly or implicitly, in the statement and answer with the position the event holds in that chain. {{start, middle, end, none}} Answer: ",
            "CCR_ECI_position_F_6": f"A causal chain is a directed path of at least three nodes in a causal graph. Answer with {{start, middle, end, none}} only. The following list of causal events [persistent barriers, region-specific barriers, economic feasibility of deploying AFOLU mitigation options, political feasibility of deploying AFOLU mitigation options] can be found in statement \"Persistent and region-specific barriers also continue to hamper the economic and political feasibility of deploying AFOLU mitigation options.\". Does the statement report a causal chain structure that includes region-specific barriers and what position does region-specific barriers hold? none. The following list of causal events [adaptation options, environmental impacts, ecosystem services, biodiversity, ecosystem resilience to climate change, adverse outcomes for different groups, inequity, maladaptive adaptation options] can be found in statement \"Adaptation options can become maladaptive due to their environmental impacts that constrain ecosystem services and decrease biodiversity and ecosystem resilience to climate change or by causing adverse outcomes for different groups, exacerbating inequity.\". Does the statement report a causal chain structure that includes inequity and what position does inequity hold? middle. The following list of causal events {event_list} can be found in statement \"{s}\". Does the statement report a causal chain structure that includes {e_i}  and what position does {e_i} hold?",
            "CCR_ECI_position_CoT_4": f"Statement: \"{s}\". All causal events in the statement: {event_list}. Based on the statement and the provided list of causal events, determine whether {e_i} is reported as part of a causal chain  and, if yes, which position it holds in that chain. A causal chain is a directed path of at least three nodes in a causal graph. Think step by step. Give reasoning and then give an answer within <Answer> {{start, middle, end, none}} </Answer>.",
            "CCR_ECI_position_CoT_5": f"Given a statement: \"{s}\" and a list of all causal events: {event_list}. Determine whether {e_i} belongs to a causal chain reported, either explicitly or implicitly, in the statement and which position it holds in that chain. A causal chain is a directed path of at least three nodes in a causal graph. Think step by step. Finally, answer with {{start, middle, end, none}}.",
            "CCR_ECI_position_CoT_6": f"The following list of causal events {event_list} can be found in statement \"{s}\". Does the statement report a causal chain structure that includes {e_i} and at which position in the chain can the event be found? A causal chain is a directed path of at least three nodes in a causal graph. Let's think step by step and answer with {{start, middle, end, none}}.",
        })

    else:
        prompts.update({
        "CCR_member_A_4": f"You will be given a causal graph. The causal relationships in this causal graph are- {G_A}. Now answer using this causal graph only, determine whether {e_i} is part of a causal chain. A causal chain is a directed path of at least three nodes in a causal graph. Think step by step. Give reasoning and then give an answer within <Answer> {{Yes, No}} </Answer>.",
        "CCR_member_A_5": f"The causal relationships in a causal graph are- {G_A}. Based on this graph, determine whether {e_i} belongs to a causal chain. A causal chain is a directed path of at least three nodes in a causal graph. Answer with {{Yes, No}} only.",
        "CCR_member_A_6": f"The given causal graph includes the following causal relations: {G_A}. Study this graph carefully, and decide whether there the graph contains a causal chain structure that includes {e_i}. Answer with {{Yes, No}} only.",
        "CCR_member_SN_4": f"You will be given a causal graph. The causal relationships in this causal graph are- {G_SN} Now answer using this causal graph only, determine whether {e_i} is part of a causal chain. A causal chain is a directed path of at least three nodes in a causal graph. Think step by step. Give reasoning and then give an answer within <Answer> {{Yes, No}} </Answer>.",
        "CCR_member_SN_5": f"The causal relationships in a causal graph are- {G_SN} Based on this graph, determine whether {e_i} belongs to a causal chain. A causal chain is a directed path of at least three nodes in a causal graph. Answer with {{Yes, No}} only.",
        "CCR_member_SN_6": f"The given causal graph includes the following causal relations: {G_SN} Study this graph carefully, and decide whether there the graph contains a causal chain structure that includes {e_i}. Answer with {{Yes, No}} only.",
        "CCR_member_ML_4": f"You will be given a causal graph. The causal relationships in this causal graph are- {G_ML}. Now answer using this causal graph only, determine whether {e_i} is part of a causal chain. A causal chain is a directed path of at least three nodes in a causal graph. Think step by step. Give reasoning and then give an answer within <Answer> {{Yes, No}} </Answer>.",
        "CCR_member_ML_5": f"The causal relationships in a causal graph are- {G_ML}. Based on this graph, determine whether {e_i} belongs to a causal chain. A causal chain is a directed path of at least three nodes in a causal graph. Answer with {{Yes, No}} only.",
        "CCR_member_ML_6": f"The given causal graph includes the following causal relations: {G_ML}. Study this graph carefully, and decide whether there the graph contains a causal chain structure that includes {e_i}. Answer with {{Yes, No}} only.",
        })

        prompts.update({
        "CCR_position_A_4": f"You will be given a causal graph. The causal relationships in this causal graph are- {G_A}. Now answer using this causal graph only, determine whether {e_i} is part of a causal chain and, if yes, which position it holds in that chain. A causal chain is a directed path of at least three nodes in a causal graph. Think step by step. Give reasoning and then give an answer within <Answer> {{start, middle, end, none}} </Answer>.",
        "CCR_position_A_5": f"The causal relationships in a causal graph are- {G_A}. Based on this graph, determine whether {e_i} belongs to a causal chain and, if yes, which position in the chain that event can be found (start, middle, or end). A causal chain is a directed path of at least three nodes in a causal graph. Answer with {{start, middle, end, none}} only.",
        "CCR_position_A_6": f"The given causal graph includes the following causal relations: {G_A}. Study this graph carefully, and decide whether there the graph contains a causal chain structure that includes {e_i} and which position {e_i} holds in that chain. Answer with {{start, middle, end, none}} only.",
        "CCR_position_SN_4": f"You will be given a causal graph. The causal relationships in this causal graph are- {G_SN}. Now answer using this causal graph only, determine whether {e_i} is part of a causal chain and, if yes, which position it holds in that chain. A causal chain is a directed path of at least three nodes in a causal graph. Think step by step. Give reasoning and then give an answer within <Answer> {{start, middle, end, none}} </Answer>.",
        "CCR_position_SN_5": f"The causal relationships in a causal graph are- {G_SN}. Based on this graph, determine whether {e_i} belongs to a causal chain and, if yes, which position in the chain that event can be found (start, middle, or end). A causal chain is a directed path of at least three nodes in a causal graph. Answer with {{start, middle, end, none}} only.",
        "CCR_position_SN_6": f"The given causal graph includes the following causal relations: {G_SN}. Study this graph carefully, and decide whether there the graph contains a causal chain structure that includes {e_i} and which position {e_i} holds in that chain. Answer with {{start, middle, end, none}} only.",
        "CCR_position_ML_4": f"You will be given a causal graph. The causal relationships in this causal graph are- {G_ML}. Now answer using this causal graph only, determine whether {e_i} is part of a causal chain and, if yes, which position it holds in that chain. A causal chain is a directed path of at least three nodes in a causal graph. Think step by step. Give reasoning and then give an answer within <Answer> {{start, middle, end, none}} </Answer>.",
        "CCR_position_ML_5": f"The causal relationships in a causal graph are- {G_ML}. Based on this graph, determine whether {e_i} belongs to a causal chain and, if yes, which position in the chain that event can be found (start, middle, or end). A causal chain is a directed path of at least three nodes in a causal graph. Answer with {{start, middle, end, none}} only.",
        "CCR_position_ML_6": f"The given causal graph includes the following causal relations: {G_ML}. Study this graph carefully, and decide whether there the graph contains a causal chain structure that includes {e_i} and which position {e_i} holds in that chain. Answer with {{start, middle, end, none}} only.",
            })

    return prompts

def obtain_graph(df):
    G = nx.DiGraph()
    for _, row in df.iterrows():
        G.add_edge(row['Cause_no_quantifier'], row['Effect_no_quantifier'])
    return G

def graph_to_graphml_string(graph: nx.DiGraph):
    """
    Convert a NetworkX graph to GraphML format as a string.

    Parameters:
        graph (nx.DiGraph): The directed graph to convert.

    Returns:
        str: GraphML representation of the graph.
    """
    buffer = io.BytesIO()
    nx.write_graphml(graph, buffer)
    return buffer.getvalue().decode('utf-8')

def graph_to_adjacency(graph: nx.DiGraph):
    edges = graph.edges()
    return ' '.join([f"({u}, {v})" for u, v in edges])

def graph_to_single_nodes(graph: nx.DiGraph):
    edges = graph.edges()
    return ' '.join([f"{u} causes {v}." for u, v in edges])

def get_events(graph: nx.DiGraph):
    return graph.nodes()

def classify_node_position_in_causal_chain(G: nx.DiGraph, node: str):
    if node not in G:
        return 'none'

    # Distances from node to descendants
    dist_from_node = nx.single_source_shortest_path_length(G, node)
    # Distances from ancestors to node (using reversed graph)
    Grev = G.reverse(copy=False)
    dist_to_node = nx.single_source_shortest_path_length(Grev, node)

    # Exclude node itself
    descendants = {n: d for n, d in dist_from_node.items() if n != node}
    ancestors = {n: d for n, d in dist_to_node.items() if n != node}

    has_descendant_dist_ge2 = any(d >= 2 for d in descendants.values())
    has_ancestor_dist_ge2 = any(d >= 2 for d in ancestors.values())

    # Classification logic
    if ancestors and descendants:
        return 'middle'
    elif descendants and not ancestors and has_descendant_dist_ge2:
        return 'start'
    elif ancestors and not descendants and has_ancestor_dist_ge2:
        return 'end'
    else:
        return 'none'


def compute_graph_metrics_with_chains(G: nx.DiGraph):
    """
    - Depth: Length of the longest directed path (in edges).
    - Breadth: Maximum out-degree of any node.
    - Node count: Total number of nodes.
    - Causal chain: Any simple directed path with at least 3 nodes (≥ 2 edges).
    """
    node_count = G.number_of_nodes()
    depth = 0
    causal_chains = []

    for node in G.nodes():
        for target in G.nodes():
            if node != target:
                for path in nx.all_simple_paths(G, source=node, target=target):
                    path_length = len(path) - 1  # edges count
                    depth = max(depth, path_length)
                    if path_length >= 2:  # at least 3 nodes
                        causal_chains.append(path)

    breadth = max(dict(G.out_degree()).values()) if node_count > 0 else 0
    num_causal_chains = len(causal_chains)

    return {
        'depth': depth,
        'breadth': breadth,
        'node_count': node_count,
        'num_causal_chains': num_causal_chains,
        'causal_chains': causal_chains
    }

def process_input_file(file_name: str,
                       cause_col: str,
                       effect_col: str,
                       task:str):
    # First load file
    df = pd.read_excel(file_name, sheet_name="causal_relations_IPCC")

    if "CCR" in task:
        results = {
            'Causal?': list(),
            'ipccText': list(),
            'depth': list(),
            'breadth': list(),
            'node_count': list(),
            'num_causal_chains': list(),
            'causal_chains': list(),
            'G_A': list(),
            'G_SN': list(),
            'G_ML': list(),
            'event': list(),
            'chain_membership_label': list(),
            'chain_position_label': list()
        }
        if "ECI" in task:
            results['event_list'] = list()
        df = df[df['Causal?'] == 'Yes']
        for statement in df['ipccText'].unique().tolist():
            statement_df = df[df['ipccText'] == statement]
            statement_graph = obtain_graph(statement_df)
            graph_metrics = compute_graph_metrics_with_chains(statement_graph)
            g_a = graph_to_adjacency(statement_graph)
            g_sn = graph_to_single_nodes(statement_graph)
            g_ml = graph_to_graphml_string(statement_graph)
            events = get_events(statement_graph)
            for event in events:
                chain_position_label = classify_node_position_in_causal_chain(statement_graph, event)
                if chain_position_label == 'none':
                    chain_membership_label = 'no'
                else:
                    chain_membership_label = 'yes'
                results['Causal?'].append('Yes')
                results['ipccText'].append(statement)
                results['depth'].append(graph_metrics['depth'])
                results['breadth'].append(graph_metrics['breadth'])
                results['node_count'].append(graph_metrics['node_count'])
                results['num_causal_chains'].append(graph_metrics['causal_chains'])
                results['causal_chains'].append(graph_metrics['causal_chains'])
                results['G_A'].append(g_a)
                results['G_SN'].append(g_sn)
                results['G_ML'].append(g_ml)
                results['event'].append(event)
                results['chain_membership_label'].append(chain_membership_label)
                results['chain_position_label'].append(chain_position_label)
                if "ECI" in task:
                    results['event_list'].append('[' + ', '.join(events) + ']')

        df = pd.DataFrame(results)

    # Add custom ids if not in df
    if 'custom_id' not in df.columns:
        df = create_custom_ids(df, file_name)

    # Add keys needed for Batch upload to OpenAI
    if "CCR" in task:
        if "ECI" in task:
            prompt_dicts = df.apply(lambda row: generate_prompt_causal_graph(task, row['event'], row['G_A'], row['G_SN'], row['G_ML'], row['ipccText'], row['event_list']), axis=1)
        else:
            prompt_dicts = df.apply(lambda row: generate_prompt_causal_graph(task, row['event'], row['G_A'], row['G_SN'], row['G_ML']), axis=1)
    else:
        prompt_dicts = df.apply(lambda row: generate_prompt(task, row[cause_col], row[effect_col], row["ipccText"]), axis=1)
    prompt_expanded = pd.DataFrame(prompt_dicts.tolist())
    full_df = pd.concat([df, prompt_expanded], axis=1)

    full_df['method'] = ["POST" for _ in range(len(full_df))]
    full_df['url'] = ["/v1/chat/completions" for _ in range(len(full_df))]

    print(full_df.iloc[-1])

    return full_df

def create_custom_ids(df, file_n):
    df['custom_id'] = ["request-" + str(i + 1) for i in range(len(df))]
    di = df.to_dict(orient='records')
    new_file_name = file_n.split(".xlsx")[0] + ".jsonl"
    with open(new_file_name, 'w') as f:
        for record in di:
            f.write(json.dumps(record) + '\n')
    return df

def process_body(prompt: str,
                 model: str):
    return {
        'model' : model,
        'messages' : [
            {
                "role": "user",
                "content": prompt
             }
        ],
        'store': True
    }

def create_batches(df: dict,
                   model_name: str,
                   task: str,
                   debugging: bool):
    if debugging:
        df = df[df['Causal?'] == 'Yes']
        new_df = df.filter(['custom_id', 'method', 'url', 'CorrI_0_1'])
        dictionary = new_df.to_dict(orient='records')
        for line in dictionary:
            line['body'] = process_body(prompt = line['CorrI_0_1'], model = model_name)
            del line['CorrI_0_1']

        batch_file_name = "Benchmarking_files/debugging.jsonl"
        directory = os.path.dirname(batch_file_name)
        # Create directories if they do not exist
        os.makedirs(directory, exist_ok=True)

        with open(batch_file_name, 'w') as f:
            for record in dictionary[:15]:
                f.write(json.dumps(record) + '\n')

    else:
        df = df[df['Causal?'] == 'Yes']
        if "CCR" in task:
            if "ECI" in task:
                all_prompting_strategies = ["_0_4", "_0_5", "_0_6", "_F_4", "_F_5", "_F_6", "_CoT_4", "_CoT_5",
                                            "_CoT_6", ]
            else:
                all_prompting_strategies = ["_A_4", "_A_5", "_A_6", "_SN_4", "_SN_5", "_SN_6", "_ML_4", "_ML_5", "_ML_6",]
        else:
            all_prompting_strategies = ["_0_1", "_0_2", "_0_3", "_F_1", "_F_2", "_F_3", "_CoT_1", "_CoT_2", "_CoT_3"]

        for strat in all_prompting_strategies:
            prompt_col = task + strat
            new_df = df.filter(['custom_id', 'method', 'url', prompt_col])
            dictionary = new_df.to_dict(orient='records')
            for line in dictionary:
                line['body'] = process_body(prompt=line[prompt_col], model=model_name)
                del line[prompt_col]

            batch_file_name = "Benchmarking_files/" + task + strat + ".jsonl"
            directory = os.path.dirname(batch_file_name)
            # Create directories if they do not exist
            os.makedirs(directory, exist_ok=True)

            with open(batch_file_name, 'w') as f:
                for record in dictionary:
                    f.write(json.dumps(record) + '\n')