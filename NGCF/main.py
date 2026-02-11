import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import time

# Importy z naszych plików
from model import NGCF
from data_utils import prepare_or_load_dataset, MovieLensTrainDataset

# --- KONFIGURACJA ---
CSV_PATH = 'archive/rating.csv' # Upewnij się, że masz ten plik
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 1024
EMB_DIM = 64
LAYERS = [64, 64] # Dwie warstwy GNN
LR = 0.001
EPOCHS = 30
DECAY = 1e-5 # L2 Regularization weight

def bpr_loss(u_emb, pos_i_emb, neg_i_emb):
    """
    Bayesian Personalized Ranking Loss
    """
    pos_scores = torch.sum(u_emb * pos_i_emb, dim=1)
    neg_scores = torch.sum(u_emb * neg_i_emb, dim=1)

    # Loss = -mean(ln(sigmoid(pos - neg)))
    loss = -torch.mean(torch.nn.functional.logsigmoid(pos_scores - neg_scores))
    return loss

def main():
    print(f"Używam urządzenia: {DEVICE}")

    # 1. Dane
    try:
        adj_matrix, train_pairs, n_users, n_items, meta = prepare_or_load_dataset(CSV_PATH)
    except FileNotFoundError:
        print(f"Błąd: Nie znaleziono pliku '{CSV_PATH}'. Pobierz MovieLens dataset.")
        return

    adj_matrix = adj_matrix.to(DEVICE)

    train_dataset = MovieLensTrainDataset(train_pairs, n_users, n_items)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    print(f"Dane gotowe. Users: {n_users}, Items: {n_items}")

    # 2. Model
    model = NGCF(n_users, n_items, emb_dim=EMB_DIM, layers=LAYERS).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=DECAY)

    # 3. Trening
    print("Rozpoczynam trening...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        start_time = time.time()

        for batch_i, (users, pos_items, neg_items) in enumerate(train_loader):
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

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1:02d}/{EPOCHS} | Loss: {avg_loss:.4f} | Time: {time.time() - start_time:.2f}s")

    # 4. Zapis
    torch.save(model.state_dict(), 'ngcf_model.pth')
    print("Model zapisany jako 'ngcf_model.pth'")

if __name__ == "__main__":
    main()