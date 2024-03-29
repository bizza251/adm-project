from dataset import GraphDataset, RandomGraphDataset, get_dataloader, get_dataset
from ils import batch_ils
from models.custom_transformer import TSPCustomTransformer, TSPTransformer
from models.utility import CustomTourLossReinforce, TourLossReinforce, TourLossReinforceMixed
from models.wrapped_models import NetworkxWrapper, RLAgentWithBaseline
from torch.optim import Adam, SGD
import torch.nn  as nn
from torch.optim.lr_scheduler import LambdaLR
from models.wrapped_models import RLAgentWithBaseline
from models.utility import TourLossReinforce
import torch
from utility import get_tour_len, iterated_local_search, logger, path_cost
import numpy as np
from functools import partial



def len_to_gt_len_ratio(model_output, batch):
    tours = model_output.tour
    tour_coords = batch.coords[torch.arange(len(tours)).view(-1, 1), tours]
    tour_len = get_tour_len(tour_coords)
    return (tour_len.cpu() / batch.gt_len).mean().item()



def valid_tour_ratio(model_output, batch):
    tours = model_output.tour
    expected_unique_nodes = tours.shape[1] - 1
    unique_nodes = torch.tensor([len(set(x.tolist())) for x in tours])
    return ((unique_nodes == expected_unique_nodes).sum() / tours.shape[0]).item()     



def avg_tour_len(model_output, batch):          
    tours = model_output.tour
    tour_coords = batch.coords[torch.arange(len(tours)).view(-1, 1), tours]
    tour_len = get_tour_len(tour_coords)
    return tour_len.mean().item()


def avg_tour_len_ils(model_output, batch, n_restarts=50, n_iterations=100, n_permutations=30, n_permutations_hillclimbing=15):
    from graph import MyGraph
    ils_results = []
    for i, g in enumerate(batch.coords):
        graph = MyGraph(coords=g)
        _, l = iterated_local_search(path_cost, graph, n_restarts, n_iterations, model_output.tour[i] + 1, \
            n_permutations, n_permutations_hillclimbing)
        ils_results.append(l)
    return np.mean(ils_results).item()



def tour_len_ils_batch(model_output, batch, n_restarts=5, n_iterations=15, k=0, max_perturbs=None):
    if k > 0:
        A = model_output.attn_matrix
        H = (- torch.log(A) * A).sum(-1)
        whitelist_idxs = H[:, 1:].topk(k, dim=-1).indices + 1
    else:
        whitelist_idxs = None
    best_tour, best_len = batch_ils(
        get_tour_len,
        batch.coords,
        model_output.tour,
        n_restarts,
        n_iterations,
        whitelist_idxs,
        max_perturbs
    )
    return best_tour, best_len


def avg_tour_len_ils_batch(model_output, batch, n_restarts=5, n_iterations=15, k=0, max_perturbs=None):
    best_tour, best_len = tour_len_ils_batch(model_output, batch, n_restarts, n_iterations, k, max_perturbs)
    return best_len.mean().item()



def load_checkpoint(path, verbose=True, **kwargs):
    checkpoint = torch.load(path, map_location='cpu')
    out = {}
    for key, obj in checkpoint.items():
        if key in kwargs:
            if hasattr(kwargs[key], 'load_state_dict'):
                kwargs[key].load_state_dict(obj)
            else:
                out[key] = obj
        else:
            out[key] = obj
            if verbose:
                logger.warning(f"`{key}` not found in checkpoint `{path}`.")
    return out



def get_model(args):
    if args.model == 'custom':
        model = TSPCustomTransformer.from_args(args)
    elif args.model == 'baseline':
        model = TSPTransformer.from_args(args)
    elif args.model == 'networkx':
        model = NetworkxWrapper()
    else:
        raise NotImplementedError()
    try:
        if args.train_mode == 'reinforce' and args.reinforce_baseline == 'baseline':
            model = RLAgentWithBaseline(model)
    except AttributeError:
        pass
    return model.to(args.device)


def get_optimizer(args, model):
    if hasattr(model, 'model'):
        params = model.model.parameters
    else:
        params = model.parameters
    if args.optimizer == 'adam':
        return Adam(params(), lr=args.learning_rate)
    elif args.optimizer == 'sgd':
        return SGD(params(), lr=args.learning_rate)
    else:
        raise NotImplementedError()


def get_train_dataset(args):
    # if args.train_dataset == 'custom':
    #     return GraphDataset()
    # else:
    #     return RandomGraphDataset(args.train_dataset)
    return RandomGraphDataset(args.train_dataset)



def get_eval_dataset(args):
    if args.train_dataset == 'custom':
        return GraphDataset()
    else:
        return RandomGraphDataset(args.eval_dataset)



def get_train_dataloader(args, dataset=None):
    if args.do_train:
        return get_dataloader(args.train_dataset, args.train_batch_size, args.dataloader_num_workers)



def get_eval_dataloader(args):
    if args.do_eval:
        return get_dataloader(args.eval_dataset, args.eval_batch_size, args.dataloader_num_workers)



def get_loss(args):
    if args.loss == 'mse':
        loss = nn.L1Loss()
    elif args.loss == 'reinforce_loss':
        # if args.model == 'custom':
        #     loss = ValidTourLossReinforce()
        # else:
        loss = TourLossReinforce()
    elif args.loss == 'reinforce_loss_mixed':
        loss = TourLossReinforceMixed()
    elif args.loss == 'custom_reinforce_loss':
        loss = CustomTourLossReinforce()
    else:
        raise NotImplementedError()
    return loss.to(args.device)


def get_transformer_lr_scheduler(optim, d_model, warmup_steps):
    for group in optim.param_groups:
        group['lr'] = 1
    def lambda_lr(s):
        d_model_ = d_model
        warm_up = warmup_steps
        s += 1
        return (d_model_ ** -.5) * min(s ** -.5, s * warm_up ** -1.5)
    return LambdaLR(optim, lambda_lr)


def get_lr_scheduler(args, optim):
    if args.lr_scheduler is None:
        return
    elif args.lr_scheduler == 'transformer':
        return get_transformer_lr_scheduler(optim, args.d_model, args.warmup_steps)
    else:
        raise NotImplementedError()



def get_metrics(args):
    metrics = {}
    if args.metrics is not None:
        for metric in args.metrics:
            if metric == 'len_to_gt_len_ratio':
                metrics[metric] = len_to_gt_len_ratio
            elif metric == 'valid_tour_ratio':
                metrics[metric] = valid_tour_ratio
            elif metric == 'avg_tour_len':
                metrics[metric] = avg_tour_len
            elif metric == 'avg_tour_len_ils':
                # f = partial(
                #     avg_tour_len_ils, 
                #     n_restarts=args.ils_n_restarts, 
                #     n_iterations=args.ils_n_iterations, 
                #     n_permutations=args.ils_n_permutations, 
                #     n_permutations_hillclimbing=args.ils_n_permutations_hillclimbing)
                f = partial(
                    avg_tour_len_ils_batch,
                    n_restarts=args.ils_n_restarts,
                    n_iterations=args.ils_n_iterations,
                    k=args.ils_k,
                    max_perturbs=args.ils_max_perturbs
                )
                metrics[metric] = f
            else:
                # TODO: eventually add other metrics
                raise NotImplementedError()
    return metrics



def get_training_commons(args):
    class dotdict(dict):
        __getattr__ = dict.get
        __setattr__ = dict.__setitem__
        __delattr__ = dict.__delitem__

    model = get_model(args)
    optimizer = get_optimizer(args, model)

    training_commons = dict(
        model=model,
        train_dataset=get_dataset(args.train_dataset),
        eval_dataset=get_dataset(args.eval_dataset),
        optimizer=optimizer,
        loss=get_loss(args),
        scheduler=get_lr_scheduler(args, optimizer),
        metrics=get_metrics(args),
        kwargs={k: v for k, v in vars(args).items() if k not in {'model', 'train_dataset', 'eval_dataset', 'optimizer', 'loss', 'epochs', 'metrics'}}
    )

    return dotdict(training_commons)