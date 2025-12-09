# ClimateCause: Complex and Implicit Causal Structures in Climate Change Reports

## Introduction

This folder contains the ClimateCause dataset, complexity metrics to measure the readability of reported causality, and code for reproducing the correlation inference and causal chain reasoning benchmarking experiments in Allein, L., Pineda-Castañeda, N., Rocci, A., Moens, M.-F. (2026): *ClimateCause: Complex and Implicit Causal Structures in Climate Change Reports*.

## ClimateCause Dataset

The dataset can be found as .xlsx file in the Data folder. The dataset schema is given in detail [below](#dataset-format)
The dataset is licensed under the [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) license.

column | type | description
--- |------| ---
`STATEMENT_LINK` | str | URL to the statement in Wikibase (no action needed)
`SECTION` | str | Section number from which the statement is taken (no action needed)
`PARAGRAPH` | str | Paragraph number from which the statement is taken (no action needed)
`SERIES_ORDINAL` | int | Position in the paragraph from which the statement is taken (no action needed)
`CONFIDENCE_LEVEL` | str | Confidence level of the statement (no action needed)
`STATEMENT` | str | The statement (no action needed)
`CAUSATION` | bool | Binary indicator (yes/no) whether the statement reports a causal relation (§\ref{causation})
`TARGET` | str | Target word(s) that evokes the causal relation (§\ref{target})
`CAUSE_NP` | str | Noun phrase reformulation of the cause (§\ref{np-formulation})
`CAUSE_CONTEXT` | str | Spatiotemporal context of the cause (§\ref{sec:contextualization_in_time_space})
`CAUSE_NO_QUANTIFIER` | str | Reformulation of the cause without quantifiers (§\ref{no-quantifiers})
`CAUSE_BELONGS_TO` | str | Event to which the cause belongs (§\ref{sec:single_events})
`EFFECT_NP` | str | Noun phrase reformulation of the effect (§\ref{np-formulation})
`EFFECT_CONTEXT` | str | Spatiotemporal context of the effect (§\ref{sec:contextualization_in_time_space})
`EFFECT_NO_QUANTIFIER` | str | Reformulation of the effect without quantifiers (§\ref{no-quantifiers})
`EFFECT_BELONGS_TO` | str | Event to which the effect belongs (§\ref{sec:single_events})
`COMBINED` | bool | Binary indicator (yes/no) whether the connection between cause/effect in `--NP` and the overarching event in `--Belongs_to` is binding
`NESTED_CAUSALITY` | bool | Binary indicator (yes/no) whether the causal relation is nested in a construction
`EXPLICITNESS` | str | Binary label (E/I) whether the causal relation is conveyed explicitly or implicitly (§\ref{explicitness})
`RELATION_TYPE` | str | Binary label (positive/negative) whether the relation type is positive (CAUSES) or negative (PREVENTS) (§\ref{relation type})
`CORRELATION` | str | Binary label (positive/negative) whether correlation is positive (increase → increase) or negative (increase → decrease) (§\ref{correlation})
`ABBREVIATIONS` | str | Set of abbreviations used in the statement resolved to their full meaning (§\ref{abbreviations})

### Requirements

This code requires Python 3.12 or higher.

Before running the code, make sure you have the following dependencies installed:

```bash
pip install -r requirements.txt
```

Prior to running ```main.py```, API keys for OpenAI and NVIDIA NIM need to obtained and included in the code. These can be applied for through their official platforms.

Adjust the implementation and evaluation settings in ```arguments.py``` before running the code. The file includes proper descriptions for each argument. For example, for reproducing the causal chain inference experiment with Deepseek R1, run the following:

```bash
python main.py --generate_chains=True --preprocessing=True --submit_to_deepseek=True
```
