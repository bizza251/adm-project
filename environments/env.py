import gym
import torch
import numpy as np
from gym import spaces
from dataset import RandomGraphDataset
from utility import BatchGraphInput, custom_collate_fn, get_tour_coords, get_tour_len



class TSPEnv(gym.Env):

    def __init__(
        self,
        dataloader,
        batch_size,
        nodes=50,
        node_features=2,
        seed=None
    ):
        super().__init__()
        self.dataloader = dataloader
        self.dataloader_it = iter(self.dataloader)
        if seed is None:
            self.seed=np.random.randint(0, int(1e6), (1,)).item()
        else:
            self.seed = seed
        self.batch_size = batch_size
        np.random.seed(seed)
        self.observation_space = gym.spaces.Box(0., 1., (batch_size, nodes, node_features), dtype=np.float32, seed=seed)
        self.action_space = gym.spaces.MultiDiscrete([nodes], dtype=np.int64, seed=seed)
        self.reset(seed)


    def _get_obs(self):
        return self.graph.coords

    
    def _get_info(self):
        return {}


    def isDone(self):
        return False


    def sample(self, n=None):
        try:
            return next(self.dataloader_it)
        except StopIteration:
            self.dataloader_it = iter(self.dataloader)
            return next(self.dataloader_it)
        

    def reset(self, seed=None, options: dict = None):
        if seed is None:
            seed = self.seed
        super().reset(seed=seed, options=options)
        self.graph = self.sample()
        return self.graph, self._get_info()


    def step(self, tour):
        tour_len = get_tour_len(get_tour_coords(self.graph.coords, tour))
        reward = self.graph.gt_len - tour_len
        self.reset()
        # obs, reward, isDone, truncated, info
        return self.graph, reward, True, False, None



def make_env(seed, dataloader, args):
    def _make_env():
        return TSPEnv(
            dataloader=dataloader, 
            batch_size=args.train_batch_size, 
            nodes=args.env_nodes,
            node_features=args.in_features,
            seed=seed)
    return _make_env



def get_env(args, dataset=None):
    if dataset is None:
        dataset = RandomGraphDataset(args.train_dataset)
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        shuffle=True,
        collate_fn=custom_collate_fn,
        drop_last=True
    )
    
    # envs = gym.vector.AsyncVectorEnv(
    #     [make_env(i, dataset, args) for i in range(args.n_envs)]
    # )
    return make_env(np.random.randint(0, 1e6, (1,)).item(), dataloader, args)()
