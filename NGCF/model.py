import torch
import torch.nn as nn
import torch.nn.functional as F

class NGCF(nn.Module):
    def __init__(self, n_users, n_items, emb_dim, layers=[64, 64], dropouts=[0.1, 0.1]):
        super(NGCF, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.emb_dim = emb_dim
        self.layer_sizes = layers
        self.dropouts = dropouts

        # Inicjalizacja osadzeń (Embeddings) dla węzłów (User + Item)
        self.embedding = nn.Embedding(n_users + n_items, emb_dim)
        self._init_weight()

        # Definicja warstw wagowych (W1, W2) dla każdej warstwy GNN
        self.gc_layers = nn.ModuleList()  # Warstwy dla W1 (liniowa transformacja)
        self.bi_layers = nn.ModuleList()  # Warstwy dla W2 (interakcja cech)

        dims = [emb_dim] + layers

        for i in range(len(layers)):
            self.gc_layers.append(nn.Linear(dims[i], dims[i+1]))
            self.bi_layers.append(nn.Linear(dims[i], dims[i+1]))

    def _init_weight(self):
        # Inicjalizacja Xaviera
        nn.init.xavier_uniform_(self.embedding.weight)

    def forward(self, adj_matrix):
        """
        adj_matrix: Sparse Tensor (Laplasjan grafu)
        """
        # Pobieramy osadzenia bazowe (warstwa 0)
        ego_embeddings = self.embedding.weight
        all_embeddings = [ego_embeddings]

        # Propagacja przez warstwy
        for i in range(len(self.layer_sizes)):
            # 1. Agregacja sąsiadów: L * E
            side_embeddings = torch.sparse.mm(adj_matrix, ego_embeddings)

            # 2. Transformacje
            # Suma: W1 * (e_i + e_u)
            sum_embeddings = self.gc_layers[i](side_embeddings + ego_embeddings)

            # Interakcja: W2 * (e_i * e_u)
            bi_embeddings = torch.mul(ego_embeddings, side_embeddings)
            bi_embeddings = self.bi_layers[i](bi_embeddings)

            # 3. Aktywacja (LeakyReLU) i Sumowanie
            ego_embeddings = nn.LeakyReLU(0.2)(sum_embeddings + bi_embeddings)

            # 4. Dropout
            ego_embeddings = F.dropout(ego_embeddings, self.dropouts[i], training=self.training)

            # 5. Normalizacja L2
            norm_embeddings = F.normalize(ego_embeddings, p=2, dim=1)
            all_embeddings.append(norm_embeddings)

        # Konkatenacja wszystkich warstw
        all_embeddings = torch.cat(all_embeddings, dim=1)

        # Rozdzielenie z powrotem na Userów i Itemów
        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)

        return u_g_embeddings, i_g_embeddings