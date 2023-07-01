import torch
from torch import nn
from torch.nn import functional as F

def feature_transform_regularizer(trans):
    d = trans.shape[1]
    I = torch.eye(d, device=trans.device)[None, :, :]
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2,1)) - I, 
                      dim=(1,2)))
    return loss

# https://github.com/pqhieu/jsis3d/blob/master/losses/discriminative.py
# fix def _variance code 
class DiscriminativeLoss(nn.Module):
    r""" Compute discriminative loss.
    Parameters
    ----------
    delta_d: float
        L_dist margin
    delta_v: float
        L_var margin
    alpha: float
        L_var weights
    beta: float
        L_dist weights
    gamma: float
        L_reg weights
    norm_p: int
        Norm in computing losses.
    """
    def __init__(self, delta_d, delta_v,
                 alpha=1.0, beta=1.0, gamma=0.001,
                 norm_p=2):
        # TODO: Respect the reduction rule
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        # Set delta_d > 2 * delta_v
        self.delta_d = delta_d
        self.delta_v = delta_v
        self.norm_p = norm_p

    def forward(self, embedded, labels):
        r"""
        Parameters
        ----------
        embedded: torch.tensor (B,C,N)
            Embeddings
        lables: torch.tensor (B,N)
            Labels
        
        Returns
        -------
        loss: torch.float
            Loss
        """
        
        embedded = embedded.transpose(1,2).contiguous()

        # arrange label numbers and get number of unique labels.
        batch_size = len(embedded)
        size = torch.zeros(batch_size, dtype=torch.long, device=embedded.device)
        for i in range(batch_size):
            unique_l = torch.unique(labels[i], return_inverse=True)
            size[i] = unique_l[0].size(0)
            labels[i] = unique_l[1]
        
        # get masks.
        masks = F.one_hot(labels)

        # get centers of embeddings
        centroids = self._centroids(embedded, masks, size)

        # get losses.
        L_v = self._variance(embedded, masks, centroids, size)
        L_d = self._distance(centroids, size)
        L_r = self._regularization(centroids, size)
        loss = self.alpha * L_v + self.beta * L_d + self.gamma * L_r

        return loss

    def _centroids(self, embedded, masks, size):
        batch_size = embedded.size(0)
        embedding_size = embedded.size(2)
        K = masks.size(2)
        x = embedded.unsqueeze(2).expand(-1, -1, K, -1)
        masks = masks.unsqueeze(3)
        x = x * masks
        centroids = []
        for i in range(batch_size):
            n = size[i]
            mu = x[i,:,:n].sum(0) / masks[i,:,:n].sum(0)
            if K > n:
                m = int(K - n)
                filled = torch.zeros(m, embedding_size)
                filled = filled.to(embedded.device)
                mu = torch.cat([mu, filled], dim=0)
            centroids.append(mu)
        centroids = torch.stack(centroids)
        # print(centroids)
        return centroids

    def _variance(self, embedded, masks, centroids, size):
        batch_size = embedded.size(0)
        num_points = embedded.size(1)
        embedding_size = embedded.size(2)
        K = masks.size(2)
        # Convert input into the same size
        mu = centroids.unsqueeze(1).expand(-1, num_points, -1, -1)
        x = embedded.unsqueeze(2).expand(-1, -1, K, -1)
        # Calculate intra pull force
        var = torch.norm(x - mu, p=self.norm_p, dim=3)
        var = torch.clamp(var - self.delta_v, min=0.0) ** 2
        var = var * masks
        loss = 0.0
        for i in range(batch_size):
            n = size[i]
            batch_loss = 0.0
            for ins_label_idx in range(n):
                batch_loss += torch.sum(var[i,:,ins_label_idx]) / torch.sum(masks[i, :, ins_label_idx])
            batch_loss /= n
            loss += batch_loss
        loss /= batch_size
        return loss

    def _distance(self, centroids, size):
        batch_size = centroids.size(0)
        loss = 0.0
        for i in range(batch_size):
            n = size[i]
            if n <= 1: continue
            mu = centroids[i, :n, :]
            mu_a = mu.unsqueeze(1).expand(-1, n, -1)
            mu_b = mu_a.permute(1, 0, 2)
            diff = mu_a - mu_b
            norm = torch.norm(diff, p=self.norm_p, dim=2)
            margin = 2 * self.delta_d * (1.0 - torch.eye(n))
            margin = margin.to(centroids.device)
            distance = torch.sum(torch.clamp(margin - norm, min=0.0) ** 2) # hinge loss
            distance /= float(n * (n - 1))
            loss += distance
        loss /= batch_size
        return loss

    def _regularization(self, centroids, size):
        batch_size = centroids.size(0)
        loss = 0.0
        for i in range(batch_size):
            n = size[i]
            mu = centroids[i, :n, :]
            norm = torch.norm(mu, self.norm_p, dim=1)
            loss += torch.mean(norm)
        loss /= batch_size
        return loss
