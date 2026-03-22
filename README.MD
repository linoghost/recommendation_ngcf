# System Rekomendacji NGCF (Neural Graph Collaborative Filtering)

Ten projekt to implementacja systemu rekomendacyjnego wykorzystującego grafowe sieci neuronowe (GCN) do przewidywania preferencji użytkowników. Model uczy się na podstawie historycznych interakcji użytkowników z filmami, tworząc wielowarstwowe reprezentacje (embeddingi) grafowe.

## 🛠 Wymagania i Instalacja

Projekt wymaga środowiska Python oraz biblioteki PyTorch z obsługą CUDA (zalecane dla wydajności).

1. Sklonuj repozytorium na swój komputer:
   ```bash
   git clone <link-do-waszego-repozytorium>

2. Zainstaluj wymagane pakiety:
    pip install -r requirements.txt

3. Pobierz zbiór danych MovieLens 20M Dataset spod linku: [text](https://www.kaggle.com/datasets/grouplens/movielens-20m-dataset)

4. W folderze NGCF utwórz nowy folder "archive" i umieśc tam plik rating.csv

5. Uruchom całość z poziomu terminala za pomocą komendy python main.py.
