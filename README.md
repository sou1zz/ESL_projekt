# Edge Benchmarking: CPU vs GPU & Model Compression

Projekt realizowany w ramach przedmiotu ESL na Akademii Górniczo Hutniczej w Krakowie.
Celem jest porównanie wydajności modelu MobileNetV2 na CPU i GPU oraz analiza wpływu kompresji modelu (pruning, kwantyzacja).
**Autorzy:**
* Karol Golański
* Tomasz Bartłomowicz
* Aleksiej Siemionow

## Opis projektu
Aplikacja przeprowadza benchmark na zbiorze danych **CIFAR-10**. Mierzymy:
* **Latency:** - czas przetwarzania obrazu [ms].
* **FPS:** - liczba klatek na sekundę.
* **Energię:** - szacowane zużycie prądu (biblioteka CodeCarbon).
* **Rozmiar modelu:** - przed i po kompresji (INT8).

## Wymagania
* Python 3.8+
* Biblioteki z pliku `requirements.txt`
* (Opcjonalnie) Karta graficzna NVIDIA z obsługą CUDA dla testów GPU.
* Wykresy zawarte w repozytorium przedstawiają wyniki testów przeprowadzone na GPU GTX1060 6GB.

## Instrukcja uruchomienia
# 1. Instalacja bibliotek
W terminalu w folderze projektu wpisz:
pip install -r requirements.txt

# 2. Uruchomienie testów
Po zainstalowaniu bibliotek w konsoli wpisujemy:
python main.py

# 3. Wygenerowanie wykresów

Gdy testy się zakończą możliwe będzie wygenerowanie wykresów. Aby to zrobić w terminalu należy wpisać:
make_plots.py
Pliki z wykresami pojawią się wtedy w naszym folderze.
