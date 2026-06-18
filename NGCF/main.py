import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import time
from tqdm import tqdm
import os
import matplotlib.pyplot as plt


from model import NGCF
from data_utils import prepare_or_load_dataset, MovieLensTrainDataset


CSV_PATH = 'archive/rating.csv' 
NGCF_PATH = 'ngcf_model.pth'
HNS_PATH = 'ngcf_model_hns.pth'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 512
EMB_DIM = 16
LAYERS = [16, 16]  #2 warswy so far
DROPOUTS = [0.1, 0.1]
LR = 0.001
EPOCHS = 40
DECAY = 1e-5

PROC_DANYCH = 0.01 #zmienna do treningu na danych, żeby nikt nie musiał czekać milion lat na model w fazach testowych

def evaluate_methods(model, adj_matrix, test_loader, train_user_dict, k=20):
    model.eval()
    hr_list = []
    mrr_list = []
    ndcg_list = []
    recall_list = []

    print(f"Używam urządzenia: {DEVICE}")
    with torch.no_grad():
        #generujemy embeddingi, mają w sobie informacje o preferencjach itd 
        u_g_embeddings, i_g_embeddings = model(adj_matrix)

        for users, pos_items in test_loader:
            users = users.to(DEVICE)
            pos_items = pos_items.to(DEVICE)

            # wyniki dla wszystkich
            # [batch_size, emb_dim] @ [emb_dim, n_items] -> [batch_size, n_items]
            scores = torch.matmul(u_g_embeddings[users], i_g_embeddings.t())

            users_list = users.cpu().tolist()
            for idx, user in enumerate(users_list):
                if user in train_user_dict:
                    train_items = train_user_dict[user]
                    # Ustawiamy wynik przedmiotów treningowych na minus nieskończoność,
                    # dzięki czemu torch.topk nigdy ich nie wybierze.
                    scores[idx, train_items] = -float('inf')

            # Pobieramy top K indeksów
            _, top_indices = torch.topk(scores, k=k)

            # hitrate
            targets = pos_items.view(-1, 1)
            hits = (top_indices == targets).any(dim=1).float()
            hr_list.extend(hits.cpu().tolist())

            # mrr
            hit_mask = (top_indices == targets)
            for row in hit_mask:
                hit_pos = torch.where(row)[0]
                if len(hit_pos) > 0:
                    mrr_list.append(1.0 / (hit_pos[0].item() + 1))
                else:
                    mrr_list.append(0.0)
            
            # ndcg
            for row in hit_mask:
                hit_pos = torch.where(row)[0]
                if len(hit_pos) > 0:
                    ndcg_list.append(1.0 / torch.log2(torch.tensor(hit_pos[0].item() + 2.0)))
                else:
                    ndcg_list.append(0.0)

            #recall
            for i in range(users.size(0)):
                current_tarets = pos_items[i].view(-1)
                hits_in_k = torch.isin(top_indices[i], current_tarets).sum().float()
                user_recall = hits_in_k / current_tarets.size(0)
                recall_list.append(user_recall.item()) 

    return sum(hr_list)/len(hr_list), sum(mrr_list)/len(mrr_list), sum(ndcg_list)/len(ndcg_list), sum(recall_list)/len(recall_list)

def bpr_loss(u_emb, pos_i_emb, neg_i_emb):
    """
    Bayesian Personalized Ranking Loss
    """
    pos_scores = torch.sum(u_emb * pos_i_emb, dim=1)
    neg_scores = torch.sum(u_emb * neg_i_emb, dim=1)

    loss = -torch.mean(torch.nn.functional.logsigmoid(pos_scores - neg_scores))
    return loss

### ----- HARD NEGATIVE SAMPLING ----- ###

# def get_hard_negatives(u_batch, i_g_embeddings, users, train_user_dict):
#     with torch.no_grad():
#         scores = torch.matmul(u_batch, i_g_embeddings.t())
#
#         users_list = users.tolist()
#         for idx, user in enumerate(users_list):
#             if user in train_user_dict:
#                 train_items = train_user_dict[user]
#                 scores[idx, train_items] = -float('inf')
#
#         _, hard_neg_indices = torch.topk(scores, k=1, dim=1) #najwyzszy wynik wsrod nieobejrzanych
#
#     return i_g_embeddings[hard_neg_indices.squeeze()] #embeddingi znalezionych hard neg

### ----- SEMI-HARD NEGATIVE SAMPLING ----- ### - Dżery - 22.05.2026

def get_hard_negatives(u_batch, i_g_embeddings, users, train_user_dict, min_rank=10, max_rank=50):
    """
    Pobiera Semi-Hard Negatives: omija `min_rank` najlepszych (zbyt ryzykowne fałszywe negatywy),
    i losuje przedmiot z przedziału od `min_rank` do `max_rank`.
    """
    with torch.no_grad():
        scores = torch.matmul(u_batch, i_g_embeddings.t())

        users_list = users.tolist()
        for idx, user in enumerate(users_list):
            if user in train_user_dict:
                train_items = train_user_dict[user]
                # Blokujemy itemy treningowe
                scores[idx, train_items] = -float('inf')

        # Zamiast brać 1 najlepszy, bierzemy paczkę (np. top 50 najlepszych)
        _, top_k_indices = torch.topk(scores, k=max_rank, dim=1)

        batch_size = u_batch.size(0)

        # Dla każdego usera losujemy pozycję rankingu w bezpiecznym przedziale (np. 10 - 49)
        # device=u_batch.device zapewnia, że tensory są na GPU jeśli tam jest model
        random_ranks = torch.randint(low=min_rank, high=max_rank, size=(batch_size,), device=u_batch.device)

        # Wyciągamy finalne ID przedmiotów z wylosowanych pozycji
        semi_hard_neg_indices = top_k_indices[torch.arange(batch_size), random_ranks]

    return i_g_embeddings[semi_hard_neg_indices]

def train_ngcf(adj_matrix, train_pairs, test_pairs, n_users, n_items, meta, train_user_dict, use_hns=True):
    print(f"Używam urządzenia: {DEVICE}")

    epoch_loses=[]

    train_dataset = MovieLensTrainDataset(train_pairs, n_users, n_items)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)

    print(f"Dane gotowe. Users: {n_users}, Items: {n_items}")

    
    model = NGCF(n_users, n_items, emb_dim=EMB_DIM, layers=LAYERS, dropouts=DROPOUTS).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=DECAY)

    
    start_time_str = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"Rozpoczynam trening... (Czas startu: {start_time_str})")
    total_batches = EPOCHS * len(train_loader)
    ema_batch_time = None
    alpha = 0.05 # Współczynnik wygładzania - im mniejszy, tym rzadsze "skoki"
    last_time = time.time()
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        start_time = time.time()

        for batch_i, (users, pos_items, neg_items) in enumerate(pbar):
            users = users.to(DEVICE)
            pos_items = pos_items.to(DEVICE)
            neg_items = neg_items.to(DEVICE)

            optimizer.zero_grad()

            # Forward pass (pełna propagacja grafu)
            u_g_embeddings, i_g_embeddings = model(adj_matrix)

            # Wybór embeddingów dla batcha
            u_batch = u_g_embeddings[users]
            pos_i_batch = i_g_embeddings[pos_items]

            if use_hns:
                neg_i_batch=get_hard_negatives(u_batch, i_g_embeddings, users, train_user_dict)
            else:
                neg_i_batch = i_g_embeddings[neg_items]

            # Loss i Backprop
            loss = bpr_loss(u_batch, pos_i_batch, neg_i_batch)
            loss.backward()

            # --- DODANE: Gradient Clipping ---
            # Nakłada limit na długość kroku, zapobiegając "eksplozji" błędu.
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            # ---------------------------------

            optimizer.step()

            #do wyswietkania remaining time
            current_time = time.time()
            batch_duration = current_time - last_time
            last_time = current_time

            
            if ema_batch_time is None:
                ema_batch_time = batch_duration
            else:
                ema_batch_time = alpha * batch_duration + (1 - alpha) * ema_batch_time
                
            batches_done = epoch * len(train_loader) + batch_i + 1
            remaining_batches = total_batches - batches_done
            
            eta_seconds = ema_batch_time * remaining_batches
            
            m, s = divmod(int(eta_seconds), 60)
            h, m = divmod(m, 60)
            
            
            red_eta_str = f"\033[91mOgólne ETA: {h:02d}:{m:02d}:{s:02d}\033[0m"
            
            pbar.set_postfix_str(f"loss: {loss.item():.4f} | {red_eta_str}")
        avg_loss = total_loss / len(train_loader)
        epoch_loses.append(avg_loss)
        print(f"Epoch {epoch+1:02d}/{EPOCHS} | Loss: {avg_loss:.4f} | Time: {time.time() - start_time:.2f}s")

    if use_hns:
        torch.save(model.state_dict(), 'ngcf_model_hns.pth')
    else:
        torch.save(model.state_dict(), 'ngcf_model.pth')

    print("Model zapisany")
    return epoch_loses


def evaluate_model(model, adj_matrix, test_pairs, n_users, n_items, train_user_dict, use_hns):
    
    test_loader = DataLoader(test_pairs, batch_size=1024, shuffle=False)

    print("Obliczanie metryk...")
    hr, mrr, ndcg, recall = evaluate_methods(model, adj_matrix, test_loader, train_user_dict, k=20)

    print(f"\nWyniki @K=20:")
    print(f"Hit Rate: {hr:.4f}")
    print(f"MRR:      {mrr:.4f}")
    print(f"NDCG:     {ndcg:.4f}")
    print(f"Recall:   {recall:.4f}")

    
    metrics = ['Hit Rate', 'MRR', 'NDCG', 'Recall']
    values = [hr, mrr, ndcg, recall]
    colors = ['#4e79a7', '#f28e2b', '#e15759', "#57e17a"]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(metrics, values, color=colors)
    
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.005, f'{yval:.4f}', ha='center', va='bottom', fontsize=12)


    prefix = ''
    if use_hns:
        prefix="HNS"
    else:
        prefix="NO_HNS"

    plt.title(f'({prefix}) Metryki Ewaluacji Modelu Rekomendacyjnego (@K=20) dla {PROC_DANYCH}%', fontsize=14)
    plt.ylabel('Wartość', fontsize=12)
    plt.ylim(0, max(values) + 0.1) 
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.savefig(f'wykresy/{prefix}_Wyniki_dla_{int(PROC_DANYCH*100)}_proc_danych.png')
    plt.show()


def plot_training_loss(epoch_losses, use_hns):
    prefix = ''
    if use_hns:
        prefix="HNS"
    else:
        prefix="NO_HNS"

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, marker='o', color='#2c3e50', linestyle='-', linewidth=2)
    
    plt.title(f'({prefix}) Krzywa uczenia (BPR Loss) dla {PROC_DANYCH} %', fontsize=14)
    plt.xlabel('Epoka', fontsize=12)
    plt.ylabel('Średni Loss', fontsize=12)
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    
    
    for i, loss in enumerate(epoch_losses):
        plt.annotate(f'{loss:.4f}', (i+1, epoch_losses[i]), textcoords="offset points", xytext=(0,10), ha='center')

    plt.tight_layout()
    plt.savefig(f'wykresy/{prefix}_Loss_plot_{int(PROC_DANYCH*100)}proc.png')
    plt.show()


def main():
    try:
        adj_matrix, train_pairs, test_pairs, n_users, n_items, meta = prepare_or_load_dataset(CSV_PATH, PROC_DANYCH)
    except FileNotFoundError:
        print(f"Błąd: Nie znaleziono pliku '{CSV_PATH}'. Pobierz MovieLens dataset.")
        return
    
    adj_matrix = adj_matrix.to(DEVICE)

    train_user_dict = {} 
    
    for pair in train_pairs:
        u = int(pair[0])
        i = int(pair[1])
        if u not in train_user_dict:
            train_user_dict[u] = []
        train_user_dict[u].append(i)

    print("Uzyc Hard negative sampling? T/N")
    hns_response = input()
    path_check = ''
    if hns_response=='N' or hns_response=='n':
        use_hns=False
        print("robimy bez")
        path_check=NGCF_PATH
    else:
        use_hns=True
        print("robimy hns")
        path_check=HNS_PATH

    if not os.path.exists(path_check):
        loses = train_ngcf(adj_matrix, train_pairs, test_pairs, n_users, n_items, meta, train_user_dict, use_hns)
        plot_training_loss(loses, use_hns)

    

    print(f"Używam urządzenia: {DEVICE}")
    
    model = NGCF(n_users, n_items, emb_dim=EMB_DIM, layers=LAYERS, dropouts=DROPOUTS)

    state_dict = torch.load(path_check, map_location=DEVICE)

    model.load_state_dict(state_dict)

    model.to(DEVICE)
    
    evaluate_model(model, adj_matrix, test_pairs, n_users, n_items, train_user_dict, use_hns)

if __name__ == "__main__":
    main()