class MyGraph(object):
    def __init__(self, name : str = None, nodes : list = None, weights : dict = None, opt : list = None, sub_opt : list = None) -> None:
        if name is not None:
            self.name = name
        if nodes is not None:
            self.nodes = nodes
        if weights is not None:
            self.edges = [k for k in weights]
            self.weights = weights
        if opt is not None:
            self.opt = opt
        if sub_opt is not None:
            self.sub_opt = sub_opt

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
        if cycle:
            for i in range(len(path) - 1):
                try:
                    cost += self.weights[(path[i],path[i+1])]
                except:
                    cost += self.weights[(path[i], path[0])]
        return cost