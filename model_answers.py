import pandas as pd
import argparse
import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

def main():
    '''
    Główna funkcja wykonawcza skryptu
    '''
    model_id, output_file, dataset_URL, max_length, use_q4, num_rows = load_parameters()
    print('Rozpoczęcie zadania...')
    df = load_dataset(dataset_URL)
    model, tokenizer = load_model_and_tokenizer(model_id, use_q4)
    pipe = create_pipeline(model, tokenizer)
    df_with_answers = get_answer(df, pipe, model_id=model_id, max_length = max_length, num_rows=num_rows)

    #Zapis wyników do pliku 
    print(f"Zapisywanie wyników do pliku {output_file}")
    df_with_answers.to_csv(output_file, index=False)

def load_dataset(dataset_URL):
    """
    Wczytuje dane z arkusza Google Sheets, usuwa nadmiarowe kolumny i filtruje wiersze z "prawidłowe" == "T".

    Args:
        dataset_url (str): URL pliku CSV.

    Returns:
        pd.DataFrame: Odfiltrowana ramka danych.
    """
    
    # Wczytanie arkusza
    df = pd.read_csv(dataset_URL)

    # Usunięcie nadmiarowych kolumn (np. 'Unnamed')
    df.drop(columns=df.columns[df.columns.str.contains('^Unnamed')], inplace=True)

    # Filtrowanie wierszy, gdzie "Prawidłowe" == "T"
    if "Prawidłowe" in df.columns:
        df = df.loc[df["Prawidłowe"] == "T"]
    else:
        raise ValueError("Brak wymaganej kolumny 'prawidłowe' w ramce danych.")
    
    return df

def load_model_and_tokenizer(model_id, use_q4):
    """
    Wczytuje model językowy i tokenizer na podstawie identyfikatora modelu.
    Obsługuje opcjonalną kwantyzację 4-bitową lub precyzję bfloat16.

    Args:
        model_id (str): Identyfikator modelu w Hugging Face.
        use_q4 (bool): Flaga określająca, czy włączyć kwantyzację 4-bitową.

    Returns:
        tuple: Załadowany model językowy i tokenizer.
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
            torch_dtype=torch.bfloat16,  # Precyzja bfloat16
            trust_remote_code=True
        )

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    return model, tokenizer

def create_pipeline(model, tokenizer):
    '''
    Tworzy pipeline dla modelu i tokenizera.
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
    Generuje odpowiedzi dla pytań w kolumnie 'Pytanie' i zapisuje je w nowej kolumnie 
    z nazwą 'Odpowiedź: model_id'.

    Args:
        df (pd.DataFrame): Ramka danych z kolumną 'Pytanie'.
        pipe: Pipeline modelu językowego.
        model_id (str): Nazwa modelu, która zostanie dodana do nagłówka kolumny odpowiedzi.
        num_rows (int, optional): Liczba wierszy do przetworzenia. Przetwarza wszystkie wiersze, jeśli None.

    Returns:
        pd.DataFrame: Zaktualizowana ramka danych z nową kolumną 'Odpowiedź: model_id'.
    """
    # Jeśli num_rows jest None, przetwarzaj wszystkie wiersze
    df_subset = df if num_rows is None else df.head(num_rows).copy()

     # Sprawdzenie, czy kolumna 'Pytanie' istnieje
    if "Pytanie" not in df.columns:
        raise ValueError("Ramka danych musi zawierać kolumnę 'Pytanie'.")
    
    # Iteracja po pytaniach i generowanie odpowiedzi
    answers = []
    for idx, question in enumerate(df_subset["Pytanie"], start=1):
        try:
            print(f"Przetwarzanie pytania {idx}/{len(df_subset)}: {question}")
            messages = [
                {"role": "user", "content": question}
            ]
            response = pipe(messages, max_length= max_length, truncation=True, do_sample=False)
            generated_text = response[0]["generated_text"]
            print(f"Odpowiedź: {generated_text}") 
        except Exception as e:
            print(f"Błąd dla pytania: {question}. Szczegóły: {e}")
            generated_text = "Błąd generacji"
        answers.append(generated_text)
    
    # Zapisanie odpowiedzi w nowej kolumnie
    answer_column_name = f"Odpowiedź: {model_id}"
    df_subset[answer_column_name] = answers
    
    return df_subset

def load_parameters():
    """
    Ładuje argumenty przekazane w wierszu poleceń.

    Returns:
        tuple: Zawiera identyfikator modelu, nazwę pliku wyjściowego, URL do datasetu
        maksymalną długość odpowiedzi, flagę kwantyzacji i ilosć wierszy.
    """
    parser = argparse.ArgumentParser(description="Skrypt generujący odpowiedzi dla benchmarku")
    parser.add_argument(
        "--model_id",
        type=str,
        required=True,
        help="Identyfikator modelu do ładowania."
    )

    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Nazwa pliku wyjściowego (CSV)."
    )

    parser.add_argument(
        "--dataset_URL",
        type= str,
        required=True,
        help= "URL zbioru danych (plik CSV)."
    )

    parser.add_argument(
        "--max_length",
        type=int,
        required= False,
        default= 1024,
        help = "Maksymalna długość wygenerowanego tekstu (w tokenach)."
    )

    parser.add_argument(
        "--use-q4",
        action="store_true",
        help="Opcjonalna flaga do włączenia kwantyzacji 4-bitowej."
    )

    parser.add_argument(
        "--num_rows",
        type= int,
        required= False,
        default= None,
        help = "Opcjonalna liczba wierszy do przetworzenia (domyślnie przertwarzane są wszystkie)."
    )

    args = parser.parse_args()
    
    return (args.model_id, args.output_file, args.dataset_URL, args.max_length, args.use_q4, args.num_rows)

if __name__ == "__main__":
    main()