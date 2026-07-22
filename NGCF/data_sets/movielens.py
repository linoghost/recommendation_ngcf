import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def load_and_process_movielens(file_path, proc_danych):
    print(f"Wczytywanie pliku CSV MovieLens: {file_path}")
    df = pd.read_csv(file_path)

    if proc_danych<1:
        print("redukcja danych o", proc_danych*100, "%")
        unique_users = df['userId'].unique()
        
        # Wybieramy losowe 25% użytkowników
        sampled_users = np.random.choice(unique_users, size=int(len(unique_users) * proc_danych), replace=False)
        
        # Zostawiamy TYLKO tych użytkowników, ale z CAŁĄ ich historią
        df = df[df['userId'].isin(sampled_users)].copy()
    
    # zostawiamy tylko oceny wieksze od 4 bo wtedy mamy takie realne zainteresowanie czyms
    df = df[df['rating'] >= 4.0].copy()

    # remapping zeby zamiast id=1928372 bylo 1, 2, 3 itd
    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()
    df['user_id_idx'] = user_encoder.fit_transform(df['userId'])
    df['item_id_idx'] = item_encoder.fit_transform(df['movieId'])

    n_users = df['user_id_idx'].nunique() #ile mamy unikalnych ziutków
    n_items = df['item_id_idx'].nunique()

    return df, n_users, n_items, user_encoder, item_encoder