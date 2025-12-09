# ClimateCause: Complex and Implicit Causal Structures in Climate Change Reports

## Introduction

This folder contains the ClimateCause dataset, complexity metrics to measure the readability of reported causality, and code for reproducing the correlation inference and causal chain reasoning benchmarking experiments in *ClimateCause: Complex and Implicit Causal Structures in Climate Change Reports*.

## ClimateCause Dataset

The dataset can be found as .xlsx file in the Data folder. The dataset schema is given in detail [below](#dataset-format)
The dataset is licensed under the [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) license.

column | type | description
--- |------| ---
`STATEMENT_LINK` | URL | URL to the statement in Wikibase (no action needed)
`SECTION` | str | Section number from which the statement is taken (no action needed)
`PARAGRAPH` | id | Paragraph number from which the statement is taken (no action needed)
`SERIES_ORDINAL` | int | Position in the paragraph from which the statement is taken (no action needed)
`CONFIDENCE_LEVEL` | str | Confidence level of the statement (no action needed)
`STATEMENT` | str | The statement (no action needed)
`CAUSATION` | bool | Binary indicator (yes/no) whether the statement reports a causal relation
`TARGET` | str | Target word(s) that evokes the causal relation
`CAUSE_NP` | str | Noun phrase reformulation of the cause
`CAUSE_CONTEXT` | str | Spatiotemporal context of the cause
`CAUSE_NO_QUANTIFIER` | str | Reformulation of the cause without quantifiers
`CAUSE_BELONGS_TO` | str | Event to which the cause belongs
`EFFECT_NP` | str | Noun phrase reformulation of the effect
`EFFECT_CONTEXT` | str | Spatiotemporal context of the effect
`EFFECT_NO_QUANTIFIER` | str | Reformulation of the effect without quantifiers
`EFFECT_BELONGS_TO` | str | Event to which the effect belongs
`COMBINED` | bool | Binary indicator (yes/no) whether the connection between cause/effect in `--NP` and the overarching event in `--Belongs_to` is binding
`NESTED_CAUSALITY` | bool | Binary indicator (yes/no) whether the causal relation is nested in a construction
`EXPLICITNESS` | str | Binary label (E/I) whether the causal relation is conveyed explicitly or implicitly
`RELATION_TYPE` | str | Binary label (positive/negative) whether the relation type is positive (CAUSES) or negative (PREVENTS)
`CORRELATION` | str | Binary label (positive/negative) whether correlation is positive (increase → increase) or negative (increase → decrease)
`ABBREVIATIONS` | str | Set of abbreviations used in the statement resolved to their full meaning

## Benchmarking: Correlation Inference and Causal Chain Reasoning

### Requirements

This code requires Python 3.13 or higher.

Before running the code, make sure you have the following dependencies installed:

```bash
pip install -r requirements.txt
```

### Complexity Metrics for Readability of Reported Causality

Run the following to **obtain the complexity measurements** for all statements in ClimateCause:

```bash
python complexity_classes.py
```

### Benchmarking: Correlation Inference and Causal Chain Reasoning

Prior to running ```main_benchmarking.py```, API keys for OpenAI  need to obtained and included in the ```submit_batch_to_openai.py```. These can be applied for through their official platforms.

Run the following to **submit** a batch file for a certain benchmarking task (e.g., CCR position) following a specific prompting variation (e.g., causal graph encoded using GraphML, third prompt) to OpenAI:

```bash
python main_benchmarking.py --preprocess_files=True --submit_to_openai=True --task="CCR_position" --prompting_strategy="_ML_6"
```

Run the following to **retrieve** a batch file for a certain benchmarking task (e.g., CCR position) following a specific prompting variation (e.g., causal graph encoded using GraphML, third prompt) from OpenAI (using a batch id) and obtain precision/recall/F1 performace results:

```bash
python main_benchmarking.py --postprocess_files=True --openai_batch_id="batch_123" --task="CCR_position" --prompting_strategy="_ML_6"
```

Once the results of all three prompt variations (e.g., "_ML_6") of a specific prompt strategy (e.g., "ML") for a benchmarking task (e.g., CCR position) are processed and saved to their respective results file, run the following to **compile the full results** (mean and standard deviation):

```bash
python main_benchmarking.py --compile_full_results=True
```

