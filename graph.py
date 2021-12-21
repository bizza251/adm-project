import torch


class MyGraph(object):
    def __init__(self, name : str = None, nodes : list = None, coords : torch.Tensor = None, weights : dict = None, opt : list = None, sub_opt : list = None, sub_opt_cost : float = None) -> None:
        if name is not None:
            self.name = name
        if nodes is not None:
            self.nodes = nodes
        if coords is not None:
            self.coords = coords
        if weights is not None:
            self.edges = [k for k in weights]
            self.weights = weights
        if opt is not None:
            self.opt = opt
        if sub_opt is not None:
            self.sub_opt = sub_opt
        if sub_opt_cost is not None:
            self.sub_opt_cost = sub_opt_cost

    def path_cost(self, path : list = None, cycle=True) -> int:
        """Compute the cost of a given path

        Args:
            path (list, optional): [description]. Defaults to None when the optimal path is considered.
            cycle (bool, optional): [description]. Defaults to True.

        Returns:
            int: total cost of the path
        """
        cost = 0
        if path is None:
            path = self.opt
        for i in range(len(path) - 1):
            try:
                cost += self.weights[(path[i],path[i+1])]
            except:
                if cycle:
                    cost += self.weights[(path[i], path[0])]
        return cost
    
    @property
    def get_opt(self):
        return torch.tensor(self.opt)
    
    @property
    def get_sub_opt(self):
        return torch.tensor(self.sub_opt)

    @property
    def get_n_nodes(self):
        return len(self.nodes)
    
    @property
    def get_edges(self):
        return torch.tensor(self.edges)