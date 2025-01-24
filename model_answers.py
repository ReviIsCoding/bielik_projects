import pandas as pd
import argparse
import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

def main():
    '''
    Główna funkcja wykonawcza skryptu
    '''
    model_id, output_file = load_parameters()
    df = load_dataset()
    model, tokenizer = load_model_and_tokenizer(model_id)
    pipe = pipe = create_pipeline(model, tokenizer)
    df_with_answers = get_answer(df, pipe, model_id=model_id, num_rows=None)

    #Zapis wyników do pliku 
    df_with_answers.to_csv(output_file, index=False)




def load_dataset():
    """
    Wczytuje dane z arkusza Google Sheets, usuwa nadmiarowe kolumny i filtruje wiersze z "prawidłowe" == "T".

    Returns:
        pd.DataFrame: Odfiltrowana ramka danych.
    """
    # URL do arkusza Google Sheets w formacie CSV
    sheet_url = "https://docs.google.com/spreadsheets/d/1-Ag5DOHUywOeg_lwDi8ymRsREmHZ87s6d6FUPfLr8tc/export?format=csv"
    
    # Wczytanie arkusza
    df = pd.read_csv(sheet_url)

    # Usunięcie nadmiarowych kolumn (np. 'Unnamed')
    df.drop(columns=df.columns[df.columns.str.contains('^Unnamed')], inplace=True)

    # Filtrowanie wierszy, gdzie "Prawidłowe" == "T"
    if "Prawidłowe" in df.columns:
        df = df.loc[df["Prawidłowe"] == "T"]
    else:
        raise ValueError("Brak wymaganej kolumny 'prawidłowe' w ramce danych.")
    
    return df

def load_model_and_tokenizer(model_id):
    """
    Wczytuje model językowy i tokenizer na podstawie identyfikatora modelu.

    Args:
        model_id (str): Identyfikator modelu w Hugging Face

    Returns:
        model: Załadowany model językowy
    """
    quantization_config = BitsAndBytesConfig(load_in_4bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        quantization_config=quantization_config,
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

def get_answer(df, pipe, model_id, num_rows = None):
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
            response = pipe(question, max_length=100, truncation=True)
            generated_text = response[0]["generated_text"]
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
        tuple: Zawiera identyfikator modelu i nazwę pliku wyjściowego.
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

    args = parser.parse_args()
    
    return (args.model_id, args.output_file)

if __name__ == "__main__":
    main()