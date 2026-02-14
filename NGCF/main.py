import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import time
from tqdm import tqdm


from model import NGCF
from data_utils import prepare_or_load_dataset, MovieLensTrainDataset


CSV_PATH = 'archive/rating.csv' 
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 1024
EMB_DIM = 64
LAYERS = [64, 64]  #2 warswy so far
LR = 0.001
EPOCHS = 30
DECAY = 1e-5 

def evaluate(model, adj_matrix, test_loader, k=20):
    model.eval()
    hr_list = []
    mrr_list = []
    ndcg_list = []

    with torch.no_grad():
        #generujemy embeddingi 
        u_g_embeddings, i_g_embeddings = model(adj_matrix)

        for users, pos_items in test_loader:
            users = users.to(DEVICE)
            pos_items = pos_items.to(DEVICE)

            # wyniki dla wszystkich
            # [batch_size, emb_dim] @ [emb_dim, n_items] -> [batch_size, n_items]
            scores = torch.matmul(u_g_embeddings[users], i_g_embeddings.t())

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

def main():
    print(f"Używam urządzenia: {DEVICE}")

    
    try:
        adj_matrix, train_pairs, n_users, n_items, meta = prepare_or_load_dataset(CSV_PATH)
    except FileNotFoundError:
        print(f"Błąd: Nie znaleziono pliku '{CSV_PATH}'. Pobierz MovieLens dataset.")
        return

    adj_matrix = adj_matrix.to(DEVICE)

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
        print(f"Epoch {epoch+1:02d}/{EPOCHS} | Loss: {avg_loss:.4f} | Time: {time.time() - start_time:.2f}s")

    
    torch.save(model.state_dict(), 'ngcf_model.pth')
    print("Model zapisany jako 'ngcf_model.pth'")

if __name__ == "__main__":
    main()