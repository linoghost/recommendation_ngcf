import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def load_and_process_yelp(file_path, proc_danych):
    print(f"Wczytywanie pliku CSV Yelp: {file_path}")
    df = pd.read_csv(file_path)

    user_col = 'user_id'
    item_col = 'business_id'
    rating_col = 'stars'

    if proc_danych < 1:
        print(f"Redukcja danych o {proc_danych*100} %")
        unique_users = df[user_col].unique()
        
        #losowe 25% userów
        sampled_users = np.random.choice(unique_users, size=int(len(unique_users)* proc_danych), replace=False)

        # Zostawiamy TYLKO tych użytkowników, ale z CAŁĄ ich historią
        df = df[df[rating_col].isin(sampled_users)].copy()

    df = df[df[rating_col] >= 4.0].copy()

    # remapping zeby zamiast id=1928372 bylo 1, 2, 3 itd
    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()
    df['user_id_idx'] = user_encoder.fit_transform(df[user_col])
    df['item_id_idx'] = item_encoder.fit_transform(df[item_col])

    n_users = df['user_id_idx'].nunique() #ile mamy unikalnych ziutków
    n_items = df['item_id_idx'].nunique()

    return df, n_users, n_items, user_encoder, item_encoder
