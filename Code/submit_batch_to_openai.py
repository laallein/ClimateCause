from openai import OpenAI

client = OpenAI(
    api_key=''
)

def create_batch(file):
    batch_input_file_id = file.id
    client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
            "description": "causal reasoning benchmarking"
        }
    )

def upload_file(jsonl_file: str):
    batch_input_file = client.files.create(
        file=open(jsonl_file, "rb"),
        purpose="batch"
    )
    return batch_input_file

def check_if_completed(bid: str):
    batch = client.batches.retrieve(bid)
    print(batch.status)
    if batch.status == 'completed':
        return batch
    else:
        print("BATCH NOT FINALIZED")
        return ""

def retrieve_results_and_save(batch_job, result_file_name):
    result_file_id = batch_job.output_file_id
    result = client.files.content(result_file_id).content
    with open(result_file_name, 'wb') as file:
        file.write(result)

    return result

def upload_to_openai(file_to_be_uploaded: str):
    batch_file = upload_file(jsonl_file=file_to_be_uploaded)
    create_batch(batch_file)
    print("File Uploaded Successfully")

def retrieve_from_openai(batch_id: str,
                         batch_results_file: str):
    batch = check_if_completed(batch_id)
    if batch != "":
        retrieve_results_and_save(batch, batch_results_file)