# ClimateCause: Complex and Implicit Causal Structures in Climate Change Reports

## Introduction

## ClimateCause Dataset

The dataset is licensed under the [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) license.

column | type | description
--- |------| ---
`ID` | int  | Id of the message extracted from Reddit
`SENTENCE` | str  | Message from which cause-effect (CE) pair is taken, with target word marked by `<t>` and `</t>` 
`CAUSATION` | str  | Target word evoking the causal relation marked by the CE pair

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
