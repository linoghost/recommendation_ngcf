import torch
import torch.utils.data as data
import pandas as pd
import numpy as np
import scipy.sparse as sp
import os
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from data_sets.movielens import load_and_process_movielens
from data_sets.yelp import load_and_process_yelp
from data_sets.amazon_books import load_and_process_amazon_books

PROCESSED_DIR = 'data_processed'

def get_dataset_loader(dataset_name, file_path, proc_danych):
    if dataset_name == 'movielens':
        return load_and_process_movielens(file_path, proc_danych)
    elif dataset_name == 'yelp':
        return load_and_process_yelp(file_path, proc_danych)
    elif dataset_name == 'amazon_books':
       return load_and_process_amazon_books(file_path, proc_danych)
    else:
        raise ValueError(f"Nieobsługiwany zbiór danych: {dataset_name}")

class MovieLensTrainDataset(data.Dataset):
    def __init__(self, train_pairs, n_users, n_items):
        self.train_pairs = train_pairs
        self.n_users = n_users
        self.n_items = n_items

        #adj list na itemy i userow
        self.train_user_set = {}
        for u, i in train_pairs:
            if u not in self.train_user_set:
                self.train_user_set[u] = set()
            self.train_user_set[u].add(i)

    def __len__(self):
        return len(self.train_pairs)

    def __getitem__(self, idx):
        user, pos_item = self.train_pairs[idx]

        # uniform negative sampling - losujemy item nieznany użytkownikowi
        neg_item = np.random.randint(0, self.n_items)
        while neg_item in self.train_user_set[user]:
            neg_item = np.random.randint(0, self.n_items)

        return user, pos_item, neg_item


def create_adj_matrix(n_users, n_items, user_item_pairs):

    # user x item
    rows = [pair[0] for pair in user_item_pairs]
    cols = [pair[1] for pair in user_item_pairs]
    data_vals = np.ones(len(rows), dtype=np.float32)

    # tworzenie macierzy interakcji (kto co obejrzał)
    R = sp.coo_matrix((data_vals, (rows, cols)), shape=(n_users, n_items)).tocsr()

    # tworzenie macierzy blokowej grafu dwudzielnego
    adj_mat = sp.bmat([[None, R], [R.T, None]], format='csr')

    # dodajemy jedynkę na przekątnej (self loops)
    adj_mat = adj_mat + sp.eye(adj_mat.shape[0], format='csr', dtype=np.float32)

    # stopień węzła (suma wierszy)
    rowsum = np.array(adj_mat.sum(1))

    # obliczanie D^-1/2 (normalizacja, żeby dany popularny film nie zakrzywiał wyników)
    d_inv = np.power(rowsum, -0.5).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat = sp.diags(d_inv, format='csr')

    # mnożenieMnożenie macierzy rzadkich: D^-1/2 * A * D^-1/2
    norm_adj = d_mat.dot(adj_mat).dot(d_mat)

    # konwersja do PyTorch Sparse (bez gęstych tablic pośrednich)
    norm_adj = norm_adj.tocoo()

    # upewniamy się, że typy danych są poprawne
    indices = np.vstack((norm_adj.row, norm_adj.col))
    values = norm_adj.data

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = norm_adj.shape

    return torch.sparse_coo_tensor(indices=i, values=v, size=shape)

def save_processed_data(dataset_name, adj_matrix, train_pairs, test_pairs, n_users, n_items, encoders):
    target_dir = os.path.join(PROCESSED_DIR, dataset_name)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    torch.save(adj_matrix, os.path.join(target_dir, 'adj_matrix.pt'))

    with open(os.path.join(target_dir, 'train_data.pkl'), 'wb') as f:
        pickle.dump(train_pairs, f)
    
    with open(os.path.join(target_dir, 'test_data.pkl'), 'wb') as f:
        pickle.dump(test_pairs, f)

    meta = {'n_users': n_users, 'n_items': n_items, 'encoders': encoders}
    with open(os.path.join(target_dir, 'meta_data.pkl'), 'wb') as f:
        pickle.dump(meta, f)
    print(f"Dane zapisane do cache.\n")


def load_processed_data(dataset_name):
    target_dir = os.path.join(PROCESSED_DIR, dataset_name)
    if not os.path.exists(os.path.join(target_dir, 'adj_matrix.pt')):
        return None

    print(f"Wczytywanie danych z cache z folderu {target_dir}...\n")
    try:
        adj_matrix = torch.load(os.path.join(target_dir, 'adj_matrix.pt'))
        with open(os.path.join(target_dir, 'train_data.pkl'), 'rb') as f:
            train_pairs = pickle.load(f)
        
        with open(os.path.join(target_dir, 'test_data.pkl'), 'rb') as f:
            test_pairs = pickle.load(f)

        with open(os.path.join(target_dir, 'meta_data.pkl'), 'rb') as f:
            meta = pickle.load(f)
        return adj_matrix, train_pairs, test_pairs, meta['n_users'], meta['n_items'], meta
    
    except Exception as e:
        print(f"Błąd cache: {e}")
        return None


def prepare_or_load_dataset(dataset_name, csv_path, proc_danych, force_rebuild=False):

    if not force_rebuild:
        data_loaded = load_processed_data(dataset_name)
        if data_loaded is not None:
            return data_loaded

    df, n_users, n_items, u_enc, i_enc = get_dataset_loader(dataset_name, csv_path, proc_danych)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    valid_users = train_df['user_id_idx'].unique()
    valid_items = train_df['item_id_idx'].unique()

    test_df = test_df[test_df['user_id_idx'].isin(valid_users)]
    test_df = test_df[test_df['item_id_idx'].isin(valid_items)]

    train_pairs = list(zip(train_df['user_id_idx'], train_df['item_id_idx']))

    test_pairs = list(zip(test_df['user_id_idx'], test_df['item_id_idx']))

    adj_matrix = create_adj_matrix(n_users, n_items, train_pairs)

    encoders = {'user': u_enc, 'item': i_enc}
    save_processed_data(dataset_name, adj_matrix, train_pairs, test_pairs, n_users, n_items, encoders)

    meta = {'n_users': n_users, 'n_items': n_items, 'encoders': encoders}
    return adj_matrix, train_pairs, test_pairs, n_users, n_items, meta