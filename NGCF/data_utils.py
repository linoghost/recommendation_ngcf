import torch
import torch.utils.data as data
import pandas as pd
import numpy as np
import scipy.sparse as sp
import os
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

PROCESSED_DIR = 'data_processed'

class MovieLensTrainDataset(data.Dataset):
    def __init__(self, train_pairs, n_users, n_items):
        self.train_pairs = train_pairs
        self.n_users = n_users
        self.n_items = n_items

        # Szybki lookup dla negative sampling
        self.train_user_set = {}
        for u, i in train_pairs:
            if u not in self.train_user_set:
                self.train_user_set[u] = set()
            self.train_user_set[u].add(i)

    def __len__(self):
        return len(self.train_pairs)

    def __getitem__(self, idx):
        user, pos_item = self.train_pairs[idx]

        # Negative Sampling
        neg_item = np.random.randint(0, self.n_items)
        while neg_item in self.train_user_set[user]:
            neg_item = np.random.randint(0, self.n_items)

        return user, pos_item, neg_item

def load_and_process_movielens(file_path):
    print(f"Wczytywanie pliku CSV: {file_path}")
    df = pd.read_csv(file_path)

    # Implicit Feedback: tylko oceny >= 4.0
    df = df[df['rating'] >= 4.0].copy()

    # Remapping ID
    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()

    df['user_id_idx'] = user_encoder.fit_transform(df['userId'])
    df['item_id_idx'] = item_encoder.fit_transform(df['movieId'])

    n_users = df['user_id_idx'].nunique()
    n_items = df['item_id_idx'].nunique()

    return df, n_users, n_items, user_encoder, item_encoder

# W pliku data_utils.py podmień funkcję create_adj_matrix na tę:

def create_adj_matrix(n_users, n_items, user_item_pairs):
    print(f"Tworzenie macierzy sąsiedztwa dla {n_users} userów i {n_items} itemów...")

    # 1. Tworzenie macierzy interakcji R (User x Item)
    rows = [pair[0] for pair in user_item_pairs]
    cols = [pair[1] for pair in user_item_pairs]
    data_vals = np.ones(len(rows), dtype=np.float32)

    # Tworzymy od razu w formacie CSR (Compressed Sparse Row)
    R = sp.coo_matrix((data_vals, (rows, cols)), shape=(n_users, n_items)).tocsr()

    # 2. Budowanie dużej macierzy A metodą blokową (bmat)
    # Zamiast alokować pustą macierz, sklejamy ją z mniejszych kawałków.
    # A = |  0   R |
    #     | R.T  0 |
    # None oznacza blok zerowy.
    adj_mat = sp.bmat([[None, R], [R.T, None]], format='csr')

    print("Normalizacja macierzy (Laplasjan)...")

    # 3. Normalizacja: D^-1/2 * (A + I) * D^-1/2

    # Dodanie self-loops (jedynki na przekątnej)
    # Używamy sp.eye w formacie CSR, żeby nie konwertować typów
    adj_mat = adj_mat + sp.eye(adj_mat.shape[0], format='csr', dtype=np.float32)

    # Obliczanie sumy wierszy
    rowsum = np.array(adj_mat.sum(1))

    # Obliczanie D^-1/2
    d_inv = np.power(rowsum, -0.5).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat = sp.diags(d_inv, format='csr')

    # Mnożenie macierzy rzadkich (bardzo szybkie i oszczędne)
    norm_adj = d_mat.dot(adj_mat).dot(d_mat)

    print("Konwersja do PyTorch Sparse Tensor...")

    # 4. Konwersja do PyTorch Sparse (bez gęstych tablic pośrednich)
    norm_adj = norm_adj.tocoo()

    # Upewniamy się, że typy danych są poprawne
    indices = np.vstack((norm_adj.row, norm_adj.col))
    values = norm_adj.data

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = norm_adj.shape

    return torch.sparse.FloatTensor(i, v, torch.Size(shape))

def save_processed_data(adj_matrix, train_pairs, n_users, n_items, encoders):
    if not os.path.exists(PROCESSED_DIR):
        os.makedirs(PROCESSED_DIR)

    torch.save(adj_matrix, os.path.join(PROCESSED_DIR, 'adj_matrix.pt'))

    with open(os.path.join(PROCESSED_DIR, 'train_data.pkl'), 'wb') as f:
        pickle.dump(train_pairs, f)

    meta = {'n_users': n_users, 'n_items': n_items, 'encoders': encoders}
    with open(os.path.join(PROCESSED_DIR, 'meta_data.pkl'), 'wb') as f:
        pickle.dump(meta, f)
    print("Dane zapisane do cache.")

def load_processed_data():
    if not os.path.exists(os.path.join(PROCESSED_DIR, 'adj_matrix.pt')):
        return None

    print("Wczytywanie danych z cache...")
    try:
        adj_matrix = torch.load(os.path.join(PROCESSED_DIR, 'adj_matrix.pt'))
        with open(os.path.join(PROCESSED_DIR, 'train_data.pkl'), 'rb') as f:
            train_pairs = pickle.load(f)
        with open(os.path.join(PROCESSED_DIR, 'meta_data.pkl'), 'rb') as f:
            meta = pickle.load(f)
        return adj_matrix, train_pairs, meta['n_users'], meta['n_items'], meta
    except Exception as e:
        print(f"Błąd cache: {e}")
        return None

def prepare_or_load_dataset(csv_path):
    data_loaded = load_processed_data()

    if data_loaded is not None:
        return data_loaded

    # Jeśli brak cache, przetwarzamy od zera
    df, n_users, n_items, u_enc, i_enc = load_and_process_movielens(csv_path)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_pairs = list(zip(train_df['user_id_idx'], train_df['item_id_idx']))
    test_pairs = list(zip(test_df['user_id_idx'], test_df['item_id_idx']))
    adj_matrix = create_adj_matrix(n_users, n_items, train_pairs)

    encoders = {'user': u_enc, 'item': i_enc}
    save_processed_data(adj_matrix, train_pairs, n_users, n_items, encoders)

    meta = {'n_users': n_users, 'n_items': n_items, 'encoders': encoders}
    return adj_matrix, train_pairs, n_users, n_items, meta