import torch
from dataset import GraphDataset, gt_matrix_from_tour
from models.custom_transformer import TSPCustomTransformer


def train(loader, model, loss, optimizer, epochs):
    for i in range(epochs):
        for sample in loader:
            n, coords, gt_tour, gt_len = sample
            gt_matrix = gt_matrix_from_tour(gt_tour[..., :-1] - 1)
            optimizer.zero_grad()
            tour, attn_matrix = model(coords.to(torch.float32))
            l = loss(attn_matrix, gt_matrix)
            l.backward()
            optimizer.step()

if __name__ == '__main__':
    dataset = GraphDataset()
    loader = torch.utils.data.DataLoader(dataset)
    model = TSPCustomTransformer(nhead = 1)
    loss = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train(loader, model, loss, optimizer, 50)