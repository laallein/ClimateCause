import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="preprocess and postprocess files for benchmarking tests")

    parser.add_argument("--debugging", type=bool, default=False, help="If true, debugging is enabled")
    parser.add_argument("--preprocess_files", type=bool, default=False, help="If true, batch files to submit to GPT5.1 are preprocessed")
    parser.add_argument("--submit_to_openai", type=bool, default=False, help="If true, submit batch files to openai server")
    parser.add_argument('--postprocess_files', type=bool, default=False, help="If true, batch results are retrieved from openai  + results are compiled")
    parser.add_argument('--compile_full_results', type=bool, default=False, help="If true, the results for all benchmark tests are compiled and shown in terminal")
    parser.add_argument('--perform_error_analysis', type=bool, default=True, help="If true, performs error analysis on the performance results")

    parser.add_argument('--openai_batch_id', type=str, default="")
    parser.add_argument("--task", type=str, default="CCR_ECI_position", choices=[
                                                                             "CorrI", "CorrI_RC",
                                                                             "CCR_member", "CCR_position",
                                                                             "CCR_ECI_member", "CCR_ECI_position"])
    parser.add_argument("--prompting_strategy", type=str, default="_ML_6", choices=[
        "_0_1", "_0_2", "_0_3",
        "_F_1", "_F_2", "_F_3",
        "_CoT_1", "_CoT_2", "_CoT_3",
        "_A_4", "_A_5", "_A_6",
        "_SN_4", "_SN_5", "_SN_6",
        "_ML_4", "_ML_5", "_ML_6",
        "_0_4", "_0_5", "_0_6",
        "_F_4", "_F_5", "_F_6",
        "_CoT_4", "_CoT_5", "_CoT_6",
    ])

    parser.add_argument("--dataset_file", type=str, default="Full_IPCC_DATASET.xlsx")
    parser.add_argument("--model_name", type=str, default="gpt-5.1-2025-11-13")
    parser.add_argument("--ECI_cause_column", type=str, default="Cause--NP")
    parser.add_argument("--ECI_effect_column", type=str, default="Effect--NP")
    parser.add_argument("--CorrI_cause_column", type=str, default="Cause_no_quantifier")
    parser.add_argument("--CorrI_effect_column", type=str, default="Effect_no_quantifier")

    parser.add_argument("--cause_column", type=str)
    parser.add_argument("--effect_column", type=str)

    parser.add_argument("--file_to_be_uploaded", type=str)

    parser.add_argument('--openai_batch_results_file', type=str)
    parser.add_argument('--results_file', type=str)
    parser.add_argument('--results_masterfile', type=str, default="Benchmarking_files/Results/results_MASTERFILE.csv")
    parser.add_argument('--label_options', type=list)
    parser.add_argument('--evaluation_target', type=str)

    args = parser.parse_args()

    args.file_to_be_uploaded = "Benchmarking_files/" + args.task + args.prompting_strategy + ".jsonl"
    if args.debugging:
        args.file_to_be_uploaded = "Benchmarking_files/debugging.jsonl"
    args.openai_batch_results_file = "Benchmarking_files/Results/" + args.task + args.prompting_strategy + "_batch.jsonl"
    args.results_file = "Benchmarking_files/Results/results_" + args.task + ".xlsx"

    if "1" in args.prompting_strategy:
        args.label_options = ['negative', 'positive']
        args.evaluation_target = "correlation"
    elif "2" in args.prompting_strategy:
        args.label_options = ['opposite', 'same']
        args.evaluation_target = "direction"
    elif "3" in args.prompting_strategy:
        args.label_options = ['decrease', 'increase']
        args.evaluation_target = "would"
    elif "member" in args.task:
        args.label_options = ['no', 'yes']
        args.evaluation_target = "Answer"
    else:
        args.label_options = ['none', 'start', 'middle', 'end']
        args.evaluation_target = "Answer"

    args.cause_column = args.CorrI_cause_column
    args.effect_column = args.CorrI_effect_column

    return args
