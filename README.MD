# System Rekomendacji NGCF (Neural Graph Collaborative Filtering)

Ten projekt to implementacja systemu rekomendacyjnego wykorzystującego grafowe sieci neuronowe (GCN) do przewidywania preferencji użytkowników. Model uczy się na podstawie historycznych interakcji użytkowników z filmami, tworząc wielowarstwowe reprezentacje (embeddingi) grafowe.

## Wymagania i Instalacja

Projekt wymaga środowiska Python oraz biblioteki PyTorch z obsługą CUDA (zalecane dla wydajności).

1. Sklonuj repozytorium na swój komputer:
   ```bash
   git clone https://github.com/linoghost/recommendation_ngcf.git
   ```
2. Utwórz i aktywuj wirtualne środowisko
    ```bash 
    python -m venv .venv
    ```
    Aktywacja na systemie Windows:
    ```bash
    .venv\Scripts\activate
    ```
    Aktywacja na Linux/Mac:
    ```bash
    source .venv/bin/activate
    ```
3. Zainstaluj wymagane pakiety:
    pip install -r requirements.txt

4. Pobierz zbiory danych:
  * Amazon Reviews 2018 - [link](https://www.kaggle.com/datasets/rogate16/amazon-reviews-2018-full-dataset)
  * MovieLens - [link](https://www.kaggle.com/datasets/grouplens/movielens-20m-dataset)
  * Yelp 2018 Reviews - [link](https://www.kaggle.com/datasets/luisfredgs/yelp-reviews-csv)

5. W folderze NGCF utwórz nowy folder "archive". W nim stwórz podfoldery "amazon_books", "movielens" oraz "yelp" i umieść odpowiednio w nich pobrane pliki.

Struktura plików powinna wyglądać tak:
```bash
NGCF/
├── archive/
│   └──amazon_books/ 
│   │   └──amazon_reviews.csv
│   │
│   └──movielens/
│   │    └──rating.csv
│   │
│   └──yelp/
│        └──yelp_review.csv
│   
├── wykresy/
├── data_utils.py
├── main.py
└── model.py
```
6. Uruchom całość z poziomu terminala za pomocą komendy python main.py.

## Uwagi:
W celu ponownego uruchomienia modelu po zmianie wartości parametrów musimy wykonać kilka niezbędnych czynności:
* Gdy zmieniamy ilość danych (`PROC_DANYCH`), to musimy usunąć folder `data_processed/`. 
  - W przeciwnym razie, skrypt załaduje stary rozmiar danych z pamięci podręcznej.

* Gdy chcemy wytrenować model od nowa, to musimy usunąć plik `ngcf_model.pth` (oraz ewentualne pliki `checkpoint.pth`). 
  - W przeciwnym razie, skrypt uzna, że model jest już gotowy, pominie całkowicie proces uczenia i przejdzie do testowania.

* Gdzie są wyniki?
  Po zakończeniu pracy skryptu, gotowe wykresy wyników oraz krzywa uczenia zapisują się w folderze `wykresy/`, a wyniki pośrednie będą pokazywały się w terminalu.
