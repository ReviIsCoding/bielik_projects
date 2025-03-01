import pandas as pd
import argparse
import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from openai import OpenAI

def main():
    '''
    The script's main function.
    '''
    model_id, output_file, dataset_URL, max_length, use_q4, num_rows, api_url, api_key = load_parameters()
    print('Getting started...')
    df = load_dataset(dataset_URL)

    if api_url:  # If a API url is specified, we use the API instead of the local model
        client = OpenAI(base_url= api_url, api_key=api_key)
        df_with_answers = get_answer_api(df, client, model_id, max_length, num_rows)
    else:
        model, tokenizer = load_model_and_tokenizer(model_id, use_q4)
        pipe = create_pipeline(model, tokenizer)
        df_with_answers = get_answer(df, pipe, model_id=model_id, max_length = max_length, num_rows=num_rows)

    # Saving to file
    print(f"Saving results to a file: {output_file}")
    df_with_answers.to_csv(output_file, index=False)

def load_dataset(dataset_URL):
    """
    Loads data from Google Sheets, removes redundant columns and filters rows with “correct” == “T”

    Args:
        dataset_url (str): URL of the CSV file.

    Returns:
        pd.DataFrame: Filtered DataFrame.
    """
    
    # Loading dataset
    df = pd.read_csv(dataset_URL)

    # Removing excessive columns (np. 'Unnamed')
    df.drop(columns=df.columns[df.columns.str.contains('^Unnamed')], inplace=True)

    # Filtering rows, where "correct" == "T"
    if "correct" in df.columns:
        df = df.loc[df["correct"] == "T"]
    else:
        raise ValueError("The required 'correct' column in the data frame is missing.")
    
    return df

def load_model_and_tokenizer(model_id, use_q4):
    """
    Loads language model and tokenizer based on model identifier.
    Supports optional 4-bit quantization or bfloat16 precision.

    Args:
        model_id (str): Hugging Face model ID.
        use_q4 (bool): Flag indicating whether to enable 4-bit quantization.

    Returns:
        tuple: Loaded language model and tokenizer.
    """
    if use_q4:
        quantization_config = BitsAndBytesConfig(load_in_4bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            quantization_config=quantization_config,
            trust_remote_code=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16,  # bfloat16 precision
            trust_remote_code=True
        )

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    return model, tokenizer

def create_pipeline(model, tokenizer):
    '''
    Creates pipeline for model and tokenizer.
    '''
    pipe = pipeline(
    "text-generation", 
    model=model, 
    tokenizer=tokenizer,
    return_full_text=False
)
    return pipe

def get_answer(df, pipe, model_id, max_length, num_rows):
    """
    Generates answers for the questions in the 'question' column and saves them in a new column 
      with the name 'generated_answer_{model_id}'.

    Args:
        df (pd.DataFrame): DataFrame with 'question' column.
        pipe: Pipeline of the language model.
        model_id (str): The name of the model, which will be added to the response column header.
        num_rows (int, optional): Number of rows to process. Processes all rows if None.

    Returns:
        pd.DataFrame: Updated data frame with new column 'generated_answer_{model_id}'.
    """
    # If num_rows is None, process all rows
    df_subset = df if num_rows is None else df.head(num_rows).copy()

     # Checking whether the 'question' column exists
    if "question" not in df.columns:
        raise ValueError("The data frame must contain 'question' column.")
    
    # Iterate through questions and generate answers
    answers = []
    for idx, question in enumerate(df_subset["question"], start=1):
        try:
            print(f"Processing question {idx}/{len(df_subset)}: {question}")
            messages = [
                {"role": "user", "content": question}
            ]
            response = pipe(messages, max_length= max_length, truncation=True, do_sample=False)
            generated_text = response[0]["generated_text"]
            print(f"Response: {generated_text}") 
        except Exception as e:
            print(f"Error for question: {question}. Szczegóły: {e}")
            generated_text = "Generation error"
        answers.append(generated_text)
    
    # Save the answer in a new column
    answer_column_name = f"generated_answer_{model_id}"
    df_subset[answer_column_name] = answers
    
    return df_subset

def get_answer_api(df, client, model_id, max_length, num_rows):
    df_subset = df if num_rows is None else df.head(num_rows).copy()

    if "question" not in df.columns:
        raise ValueError("The data must contain 'question' column.")
    answers = []
    for idx, question in enumerate(df_subset["question"], start = 1):
        try:
            print(f"Processing question {idx}/{len(df_subset)}: {question}")
            response = client.chat.completions.create(
                model = model_id,
                messages = [{'role':'user', 'content':question,}],
                max_tokens = max_length
            )
            generated_text = response.choices[0].message.content.strip()
            print(f"Response: {generated_text}")
        except Exception as e:
            print(f"Error for question: {question}. Details: {e}")
            generated_text = "Generation error"
        answers.append(generated_text)

        df_subset[f"generated_answer_{model_id}"] = answers
    return df_subset

def load_parameters():
    """
    Loads arguments passed on command line.

    Returns:
        tuple: Contains model identifier, output file name, URL to dataset, maximum response length, 
        quantization flag, number of rows, API url and API key.
    """
    parser = argparse.ArgumentParser(description="Script that generates answers for the benchmark")
    parser.add_argument(
        "--model_id",
        type=str,
        required=True,
        help="Model ID to be loaded."
    )

    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Output file name (CSV)."
    )

    parser.add_argument(
        "--dataset_URL",
        type= str,
        required=True,
        help= "Dataset URL (CSV file)."
    )

    parser.add_argument(
        "--max_length",
        type=int,
        required= False,
        default= 1024,
        help = "Maximum length of the generated text (in tokens)."
    )

    parser.add_argument(
        "--use-q4",
        action="store_true",
        help="Optional flag to enable 4-bit quantization"
    )

    parser.add_argument(
        "--num_rows",
        type= int,
        required= False,
        default= None,
        help = "Optional number of rows to process (by default all rows are processed)."
    )
    parser.add_argument(
        "--api_url", 
        type=str, 
        default=None, 
        help="Base URL of the API endpoint."
        )
    
    parser.add_argument(
        "--api_key", 
        type=str, 
        default=None, 
        help="API key for authentication."
        )

    args = parser.parse_args()
    
    #Validate num_rows value
    if args.num_rows is not None and args.num_rows <= 0:
        raise argparse.ArgumentTypeError("The value of num_rows must be greater than 0.")

    return (args.model_id, args.output_file, args.dataset_URL, args.max_length, args.use_q4, args.num_rows, args.api_url, args.api_key)

if __name__ == "__main__":
    main()