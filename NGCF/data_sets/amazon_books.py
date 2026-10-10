import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def load_and_process_amazon_books(file_path, proc_danych):
    print(f"Wczytywanie pliku CSV AmazonBooks: {file_path}")
    df = pd.read_csv(file_path)

    user_col = 'userName'
    item_col = 'itemName'
    rating_col = 'rating'

    #wymuszamy by oceny były traktowane jako liczby
    df[rating_col] = pd.to_numeric(df[rating_col], errors='coerce')

    #odrzucamy wersję bez ocen
    df = df.dropna(subset=[user_col,item_col,rating_col])

    if proc_danych < 1:
        print(f"Redukcja danych o {proc_danych*100} %")
        unique_users = df[user_col].unique()
        
        sampled_users = np.random.choice(unique_users, size=max(1, int(len(unique_users)* proc_danych)), replace=False)
        
        # Zostawiamy TYLKO tych użytkowników, ale z CAŁĄ ich historią
        df = df[df[user_col].isin(sampled_users)].copy()

    df = df[df[rating_col] >= 4.0].copy()

    if len(df) == 0:
        raise ValueError("Po odruceniu ocen < 4.0 zbiór danych jest pusty. Zwiększ PROC_DANYCH.")

    # remapping zeby zamiast id=1928372 bylo 1, 2, 3 itd
    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()
    df['user_id_idx'] = user_encoder.fit_transform(df[user_col])
    df['item_id_idx'] = item_encoder.fit_transform(df[item_col])

    n_users = df['user_id_idx'].nunique() #ile mamy unikalnych userów
    n_items = df['item_id_idx'].nunique()

    return df, n_users, n_items, user_encoder, item_encoder