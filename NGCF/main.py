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
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 1024
EMB_DIM = 64
LAYERS = [64, 64]  #2 warswy so far
LR = 0.001
EPOCHS = 30
DECAY = 1e-5 

PROC_DANYCH = 0.15 #zmienna do treningu na danych, żeby nikt nie musiał czekać milion lat na model
#w fazach testowych

def evaluate_methods(model, adj_matrix, test_loader, train_user_dict, k=20):
    model.eval()
    hr_list = []
    mrr_list = []
    ndcg_list = []
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

    return sum(hr_list)/len(hr_list), sum(mrr_list)/len(mrr_list), sum(ndcg_list)/len(ndcg_list)

def bpr_loss(u_emb, pos_i_emb, neg_i_emb):
    """
    Bayesian Personalized Ranking Loss
    """
    pos_scores = torch.sum(u_emb * pos_i_emb, dim=1)
    neg_scores = torch.sum(u_emb * neg_i_emb, dim=1)

   
    loss = -torch.mean(torch.nn.functional.logsigmoid(pos_scores - neg_scores))
    return loss


def train_ngcf(adj_matrix, train_pairs, test_pairs, n_users, n_items, meta):
    print(f"Używam urządzenia: {DEVICE}")

    epoch_loses=[]

    train_dataset = MovieLensTrainDataset(train_pairs, n_users, n_items)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    print(f"Dane gotowe. Users: {n_users}, Items: {n_items}")

    
    model = NGCF(n_users, n_items, emb_dim=EMB_DIM, layers=LAYERS).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=DECAY)

    
    print("Rozpoczynam trening...")
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
            neg_i_batch = i_g_embeddings[neg_items]

            # Loss i Backprop
            loss = bpr_loss(u_batch, pos_i_batch, neg_i_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            pbar.set_postfix({'loss': loss.item()})

        avg_loss = total_loss / len(train_loader)
        epoch_loses.append(avg_loss)
        print(f"Epoch {epoch+1:02d}/{EPOCHS} | Loss: {avg_loss:.4f} | Time: {time.time() - start_time:.2f}s")

    
    torch.save(model.state_dict(), 'ngcf_model.pth')
    print("Model zapisany jako 'ngcf_model.pth'")
    return epoch_loses


def evaluate_model(model, adj_matrix, test_pairs, n_users, n_items, train_user_dict):
    
    test_loader = DataLoader(test_pairs, batch_size=1024, shuffle=False)

    print("Obliczanie metryk...")
    hr, mrr, ndcg = evaluate_methods(model, adj_matrix, test_loader, train_user_dict, k=20)

    print(f"\nWyniki @K=20:")
    print(f"Hit Rate: {hr:.4f}")
    print(f"MRR:      {mrr:.4f}")
    print(f"NDCG:     {ndcg:.4f}")

    
    metrics = ['Hit Rate', 'MRR', 'NDCG']
    values = [hr, mrr, ndcg]
    colors = ['#4e79a7', '#f28e2b', '#e15759']

    plt.figure(figsize=(10, 6))
    bars = plt.bar(metrics, values, color=colors)
    
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.005, f'{yval:.4f}', ha='center', va='bottom', fontsize=12)

    plt.title('Metryki Ewaluacji Modelu Rekomendacyjnego (@K=20)', fontsize=14)
    plt.ylabel('Wartość', fontsize=12)
    plt.ylim(0, max(values) + 0.1) 
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.savefig(f'wykresy/Wyniki_dla_{int(PROC_DANYCH*100)}_proc_danych.png')
    plt.show()


def plot_training_loss(epoch_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, marker='o', color='#2c3e50', linestyle='-', linewidth=2)
    
    plt.title('Krzywa uczenia (BPR Loss)', fontsize=14)
    plt.xlabel('Epoka', fontsize=12)
    plt.ylabel('Średni Loss', fontsize=12)
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    
    
    for i, loss in enumerate(epoch_losses):
        plt.annotate(f'{loss:.4f}', (i+1, epoch_losses[i]), textcoords="offset points", xytext=(0,10), ha='center')

    plt.tight_layout()
    plt.savefig(f'wykresy/Loss_plot_{int(PROC_DANYCH*100)}proc.png')
    plt.show()


def main():
    try:
        adj_matrix, train_pairs, test_pairs, n_users, n_items, meta = prepare_or_load_dataset(CSV_PATH, PROC_DANYCH)
    except FileNotFoundError:
        print(f"Błąd: Nie znaleziono pliku '{CSV_PATH}'. Pobierz MovieLens dataset.")
        return
    
    adj_matrix = adj_matrix.to(DEVICE)

    if not os.path.exists(NGCF_PATH):
        loses = train_ngcf(adj_matrix, train_pairs, test_pairs, n_users, n_items, meta)
        plot_training_loss(loses)

    train_user_dict = {} #robię słownik dla rzeczy ktore widzial uzytkownik
    #nie jestem pewna, czy to jest najlepszy sposób na poprawienie modelu
    #ale w evaluation methods model chyba bierze tylko dane z treningu 
    # jako wyniki for some reason
    
    for pair in train_pairs:
        u = int(pair[0])
        i = int(pair[1])
        if u not in train_user_dict:
            train_user_dict[u] = []
        train_user_dict[u].append(i)

    print(f"Używam urządzenia: {DEVICE}")
    
    model = NGCF(n_users, n_items, emb_dim=EMB_DIM, layers=LAYERS)

    state_dict = torch.load(NGCF_PATH, map_location=DEVICE)

    model.load_state_dict(state_dict)

    model.to(DEVICE)
    
    evaluate_model(model, adj_matrix, test_pairs, n_users, n_items, train_user_dict)




if __name__ == "__main__":
    main()