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

        
        self.embedding = nn.Embedding(n_users + n_items, emb_dim)
        self._init_weight()

        
        self.gc_layers = nn.ModuleList()  
        self.bi_layers = nn.ModuleList()  

        dims = [emb_dim] + layers

        for i in range(len(layers)):
            self.gc_layers.append(nn.Linear(dims[i], dims[i+1]))
            self.bi_layers.append(nn.Linear(dims[i], dims[i+1]))

    def _init_weight(self):
        
        nn.init.xavier_uniform_(self.embedding.weight)

    def forward(self, adj_matrix):
        """
        adj_matrix: Sparse Tensor (Laplasjan grafu)
        """
        
        ego_embeddings = self.embedding.weight
        all_embeddings = [ego_embeddings]

       
        for i in range(len(self.layer_sizes)):
            # agregacja sasiadow
            side_embeddings = torch.sparse.mm(adj_matrix, ego_embeddings)

            # transformacje
            # suma: W1 * (e_i + e_u)
            sum_embeddings = self.gc_layers[i](side_embeddings + ego_embeddings)

            # interakcja: W2 * (e_i * e_u)
            bi_embeddings = torch.mul(ego_embeddings, side_embeddings)
            bi_embeddings = self.bi_layers[i](bi_embeddings)

            # aktywacja i sumowanie
            ego_embeddings = nn.LeakyReLU(0.2)(sum_embeddings + bi_embeddings)

            # dropout
            ego_embeddings = F.dropout(ego_embeddings, self.dropouts[i], training=self.training)

            # normalizaja
            norm_embeddings = F.normalize(ego_embeddings, p=2, dim=1)
            all_embeddings.append(norm_embeddings)

        
        all_embeddings = torch.cat(all_embeddings, dim=1)

        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)

        return u_g_embeddings, i_g_embeddings