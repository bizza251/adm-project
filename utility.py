def path_cost(path : list, weights : dict, cycle=True) -> int:
    """Computes the cost of the given path

    Args:
        path (list): tour taken\n
        weights (dict): key -> edge : value -> weight.\n
        cycle (bool, optional): [description]. Defaults to True.\n

    Returns:
        int: final cost
    """
    cost = 0
    if cycle:
        for i in range(len(path) - 1):
            try:
                cost += weights[(path[i],path[i+1])]
            except:
                cost += weights[(path[i], path[0])]
    return cost