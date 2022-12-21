import torch
from dataset import GraphDataset, gt_matrix_from_tour
from models.custom_transformer import TSPCustomTransformer
import logging
from tqdm import tqdm
from torch.optim import Adam, SGD
import torch.nn  as nn

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(
    format="%(levelname)s: %(message)s"
)

def get_model(args):
    return TSPCustomTransformer.from_args(args).to(args.device)

def get_optimizer(args, model):
    if args.optimizer == 'adam':
        return Adam(model.parameters(), lr=args.learning_rate)
    elif args.optimizer == 'sgd':
        return SGD(model.parameters(), lr=args.learning_rate)
    else:
        raise NotImplementedError()

def get_train_dataset(args):
    return GraphDataset()

def get_eval_dataset(args):
    return GraphDataset()

def get_train_dataloader(args):
    dataset = get_train_dataset(args)
    return torch.utils.data.DataLoader(dataset, batch_size=args.batch_size)

def get_eval_dataloader(args):
    if args.do_eval:
        dataset = get_eval_dataset(args)
        return torch.utils.data.DataLoader(dataset, batch_size=args.batch_size)
    return None

def get_loss(args):
    if args.loss == 'mse':
        loss = nn.MSELoss()
    else:
        raise NotImplementedError()
    return loss.to(args.device)

def get_transformer_lr_scheduler(optim, d_model, warmup_steps):
    for group in optim.param_groups:
        group['lr'] = 1
    def lambda_lr(s):
        d_model = d_model
        warm_up = warmup_steps
        s += 1
        return (d_model ** -.5) * min(s ** -.5, s * warm_up ** -1.5)
    return LambdaLR(optim, lambda_lr)

def get_lr_scheduler(args, optim):
    if args.lr_scheduler == 'transformer':
        return get_transformer_lr_scheduler(optim, args.d_model, args.warmup_steps)
    else:
        raise NotImplementedError()


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



def load_checkpoint(path, **kwargs):
    checkpoint = torch.load(path)
    out = {}
    for key, obj in kwargs.items():
        if key in checkpoint:
            if hasattr(obj, 'load_state_dict'):
                obj.load_state_dict(checkpoint[key])
            else:
                out[key] = checkpoint[key]
        else:
            logger.warning(f"`{key}` not found in checkpoint `{path}`.")
    return out



class Trainer:

    def __init__(
        self,
        model,
        train_dataloader,
        optimizer,
        loss,
        epochs,
        eval_dataloader=None,
        scheduler=None,
        checkpoint_path=None,
        resume_from_checkpoint=None,
        device='cpu',
        metrics=None,
        *args,
        **kwargs
    ):

        self.model = model
        self.train_dataloader = train_dataloader
        self.optimizer = optimizer
        self.loss = loss
        self.epochs = epochs
        self.eval_dataloader = eval_dataloader
        self.scheduler = scheduler
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.metrics = metrics

        self.best_loss = torch.inf
        self.best_metrics = {}
        self.start_epoch = 0

        if resume_from_checkpoint:
            logger.info(f"Resuming from checkpoint `{resume_from_checkpoint}`...")
            checkpoint_data = load_checkpoint(resume_from_checkpoint)
            if 'epoch' in checkpoint_data:
                self.start_epoch = checkpoint_data['epoch']
                del checkpoint_data['epoch']
            for k, v in vars(self):
                if k in checkpoint_data:
                    setattr(self, k, checkpoint_data[k])
            logger.info("Checkpoint loaded!")

    
    @classmethod
    def from_args(cls, args):
        kwargs = {k: v for k, v in vars(args).items() if k not in {'model', 'train_dataloader', 'eval_dataloader', 'optimizer', 'loss', 'epochs'}}
        model = get_model(args)
        train_dataloader = get_train_dataloader(args)
        eval_dataloader = get_eval_dataloader(args)
        optimizer = get_optimizer(args, model)
        loss = get_loss(args)

        return cls(
            model,
            train_dataloader,
            optimizer,
            loss,
            args.epochs,
            eval_dataloader,
            **kwargs)


    def train_step(self, batch):
        '''Change this method with MethodType to customize behavior.'''
        n_nodes, coords, gt_tours, gt_lens = batch
        if coords.device != self.device:
            coords = coords.to(self.device)
        if not torch.is_floating_point(coords):
            coords = coords.to(torch.float32)
        gt_matrices = gt_matrix_from_tour(gt_tours[..., :-1] - 1).to(self.device)
        self.optimizer.zero_grad()
        tours, attn_matrices = self.model(coords)
        # model output, gt
        l = self.loss(attn_matrices, gt_matrices)
        self.optimizer.step()
        return l

    
    def eval_step(self, batch):
        '''Change this method with MethodType to customize behavior.'''
        n_nodes, coords, gt_tours, gt_lens = batch
        if coords.device != self.device:
            coords = coords.to(self.device)
        if not torch.is_floating_point(coords):
            coords = coords.to(torch.float32)
        gt_matrices = gt_matrix_from_tour(gt_tours[..., :-1] - 1).to(self.device)
        tours, attn_matrices = self.model(coords)
        l = self.loss(attn_matrices, gt_matrices)
        if self.metrics:
            metrics_results = {}
            for metric_name, metric_fun in self.metrics.items():
                # model output, gt
                metrics_results[metric_name] = metric_fun((tours, attn_matrices), (gt_matrices, gt_tours))
        return l, metrics_results

    
    def do_train(self):
        self.model.train()
        for epoch in range(self.start_epoch, self.epochs):
            epoch_loss = 0
            n_samples = 0
            # for i, batch in tqdm(enumerate(self.train_dataloader), desc=f"Epoch {epoch}/{self.epochs}"):
            for i, batch in enumerate(tqdm(self.train_dataloader, desc=f"Epoch {epoch}/{self.epochs}")):
                step_loss = self.train_step(batch)
                epoch_loss += step_loss
                n_samples += len(batch)
            
            # TODO: log to tensorboard
            if n_samples:
                epoch_loss /= n_samples
                logger.info(f"[epoch {epoch}] loss: {epoch_loss} | Min loss: {self.best_loss}")
                if epoch_loss < self.best_loss:
                    logger.info(f"[epoch {epoch}] New min loss: {epoch_loss}")
                    self.best_loss = epoch_loss

            # TODO: run evaluation
            # TODO: save checkpoint
        logger.info("Training completed!")



from torch.optim.lr_scheduler import _LRScheduler, LambdaLR


def get_model(args):
    return TSPCustomTransformer.from_args(args).to(args.device)

def get_optimizer(args, model):
    if args.optimizer == 'adam':
        return Adam(model.parameters(), lr=args.learning_rate)
    elif args.optimizer == 'sgd':
        return SGD(model.parameters(), lr=args.learning_rate)
    else:
        raise NotImplementedError()

def get_train_dataset(args):
    return GraphDataset()

def get_eval_dataset(args):
    return GraphDataset()

def get_train_dataloader(args):
    dataset = get_train_dataset(args)
    return torch.utils.data.DataLoader(dataset, batch_size=args.batch_size)

def get_train_dataloader(args):
    dataset = get_eval_dataset(args)
    return torch.utils.data.DataLoader(dataset, batch_size=args.batch_size)

def get_loss(args):
    if args.loss == 'mse':
        loss = nn.MSELoss()
    else:
        raise NotImplementedError()
    return loss.to(args.device)

def get_transformer_lr_scheduler(optim, d_model, warmup_steps):
    for group in optim.param_groups:
        group['lr'] = 1
    def lambda_lr(s):
        d_model = d_model
        warm_up = warmup_steps
        s += 1
        return (d_model ** -.5) * min(s ** -.5, s * warm_up ** -1.5)
    return LambdaLR(optim, lambda_lr)

def get_lr_scheduler(args, optim):
    if args.lr_scheduler == 'transformer':
        return get_transformer_lr_scheduler(optim, args.d_model, args.warmup_steps)
    else:
        raise NotImplementedError()


if __name__ == '__main__':
    dataset = GraphDataset()
    loader = torch.utils.data.DataLoader(dataset, batch_size=4)
    model = TSPCustomTransformer(nhead = 4)
    loss = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train(loader, model, loss, optimizer, 50)