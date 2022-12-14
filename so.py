import torch
import random


class EchoDataset(torch.utils.data.IterableDataset):
    
  def __init__(self, delay=4, seq_length=15, size=1000):
    super(EchoDataset, self).__init__()
    self.delay = delay
    self.seq_length = seq_length
    self.size = size
  
  def __len__(self):
    return self.size

  def __iter__(self):
    """ Iterable dataset doesn't have to implement __getitem__.
        Instead, we only need to implement __iter__ to return
        an iterator (or generator).
    """
    # for _ in range(self.size):
    #   seq = torch.tensor([random.choice(range(1, N + 1)) for i in range(self.seq_length)], dtype=torch.int64)
    #   result = torch.cat((torch.zeros(self.delay), seq[:self.seq_length - self.delay])).type(torch.int64)
    #   yield seq, result
    seq = torch.tensor([random.choice(range(1, N + 1)) for i in range(self.seq_length)], dtype=torch.int64)
    result = torch.cat((torch.zeros(self.delay), seq[:self.seq_length - self.delay])).type(torch.int64)
    return seq, result


N = 10
DELAY = 4
DATASET_SIZE = 200000

if __name__ == '__main__':
    ds = EchoDataset(delay=DELAY, size=DATASET_SIZE)

    train_count = int(0.7 * DATASET_SIZE)
    valid_count = int(0.2 * DATASET_SIZE)
    test_count = DATASET_SIZE - train_count - valid_count
    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
        ds, (train_count, valid_count, test_count)
    )
    train_dataset_loader = torch.utils.data.DataLoader(train_dataset)
    # iterator = iter(train_dataset_loader)
    # print(next(iterator))
    [print(item) for item in train_dataset_loader]

