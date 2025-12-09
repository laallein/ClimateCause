from arguments_benchmarking import parse_args
from create_batch_files import process_input_file, create_batches
from submit_batch_to_openai import upload_to_openai, retrieve_from_openai
from postprocessing_results import postprocessing, compile_results
from error_analysis import perform_error_analysis
import os

def main():
    args = parse_args()

    if args.preprocess_files:
        f = process_input_file(
            file_name=args.dataset_file,
            cause_col=args.cause_column,
            effect_col=args.effect_column,
            task=args.task
            )

        f.to_excel(args.task+ "_" + args.dataset_file)

        create_batches(
            df=f,
            model_name=args.model_name,
            task=args.task,
            debugging=args.debugging
        )

    if args.submit_to_openai:
        print("Submitting to OpenAI...")
        upload_to_openai(file_to_be_uploaded=args.file_to_be_uploaded)

    elif args.postprocess_files:
        print("Postprocessing files...")
        if not os.path.isfile(args.openai_batch_results_file):
            retrieve_from_openai(batch_id=args.openai_batch_id,
                                 batch_results_file=args.openai_batch_results_file)
        postprocessing(task=args.task,
                       prompting_strategy=args.prompting_strategy,
                       batch_file=args.openai_batch_results_file,
                       dataset_file=args.dataset_file,
                       results_file=args.results_file,
                       results_masterfile=args.results_masterfile,
                       labels=args.label_options,
                       target=args.evaluation_target)

    elif args.compile_full_results:
        compile_results(f=args.results_masterfile)

    elif args.perform_error_analysis:
        perform_error_analysis(results_file=args.results_file)

if __name__ == "__main__":
    main()
