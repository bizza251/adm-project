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
    for i in range(len(path) - 1):
        try:
            cost += weights[(path[i],path[i+1])]
        except:
            if cycle:
                cost += weights[(path[i], path[0])]
                pass
    return cost

def read_file_from_directory(path, type = None, absolute=False):
    import os
    if not absolute:
        filename = []
        for file in os.listdir(path):
            if type != None and file.split('.')[-1] == type:
                filename.append(file)
    else:
        filename = {file : os.path.join(path, file) for file in os.listdir(path) if type != None and file.split('.')[-1] == type}
    return filename