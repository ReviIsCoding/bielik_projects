Wczytaj plik z pytaniami do ramki danych Pandas
Każde pytanie jako osobny wiersz. Wyczyć ramkę z niepotrzebnych kolumn.

Dla każdego zapytania w ramce danych:
    Wyślij zapytanie do API Huuging Face (Bielik-11B-v2.3-Instruct)
    Otzymaj odpowiedź modelu.
    Zapisz odpowiedź w ramce danych w nowej kolumnie.

Dla każdego zapytania w ramce danych:
    Wykonaj zapytanie do Google/Bing
    Pobierz listę top X wyników (np 5 linków)
    Zapisz znalezione URL w ramce danych (np jako listę "Źródła").

Dla każdego URL w wynikach wyszukiwania:
    Pobierz zawartość strony (requests lub Selenium)
    Przetwórz HTML (BeautifulSoup, usunięcie nieistotnych treści)
    Wydobądź główną treść artykułu
    Zapisz tekst jako odpowiedż "z internetu" w ramce danych

Zapisz ramkę danych do pliku CSV


Potrzebne biblioteki:
pandas
requests / Selenium
BeautifulSoup
OpenAI
time
transformers (?)
coś do wyszykiwania w sieci po API