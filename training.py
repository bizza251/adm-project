import pathlib
import torch
from dataset import GraphDataset, gt_matrix_from_tour
from models.custom_transformer import TSPCustomTransformer
import logging
from tqdm import tqdm
from torch.optim import Adam, SGD
import torch.nn  as nn
import os


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
    for key, obj in checkpoint.items():
        if key in kwargs:
            if hasattr(kwargs[key], 'load_state_dict'):
                kwargs[key].load_state_dict(obj)
            else:
                out[key] = obj
        else:
            out[key] = obj
            logger.warning(f"`{key}` not found in checkpoint `{path}`.")
    return out



class Trainer:

    exclude_from_checkpoint = {
        'train_dataloader',
        'save_epochs',
        'epochs',
        'eval_dataloader',
        'checkpoint_dir',
        'metrics',
        'loss',
        'device'
    }

    def __init__(
        self,
        model,
        train_dataloader,
        optimizer,
        loss,
        epochs,
        eval_dataloader=None,
        scheduler=None,
        checkpoint_dir=None,
        resume_from_checkpoint=None,
        device='cpu',
        metrics=None,
        save_epochs=5,
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
        self.checkpoint_dir = checkpoint_dir
        self.device = device
        self.metrics = metrics
        self.save_epochs = save_epochs

        if checkpoint_dir:
            pathlib.Path(checkpoint_dir).mkdir(parents=True, exist_ok=True) 

        self.best_loss = torch.inf
        self.best_metrics = {}
        self.start_epoch = 0

        if resume_from_checkpoint:
            logger.info(f"Resuming from checkpoint `{resume_from_checkpoint}`...")
            attrs = {k: v for k, v in vars(self).items() if k not in self.exclude_from_checkpoint}
            checkpoint_data = load_checkpoint(resume_from_checkpoint, **attrs)
            if 'epoch' in checkpoint_data:
                self.start_epoch = checkpoint_data['epoch'] + 1
                checkpoint_data.pop('epoch', None)
                checkpoint_data.pop('epochs', None)
                checkpoint_data.pop('start_epoch', None)
            for k, v in attrs.items():
                if k in checkpoint_data:
                    setattr(self, k, checkpoint_data[k])
            logger.info("Checkpoint loaded!")

        self.model.to(device)
        self.loss.to(device)

    
    @classmethod
    def from_args(cls, args):
        kwargs = {k: v for k, v in vars(args).items() if k not in {'model', 'train_dataloader', 'eval_dataloader', 'optimizer', 'loss', 'epochs'}}
        model = get_model(args)
        train_dataloader = get_train_dataloader(args)
        eval_dataloader = get_eval_dataloader(args)
        optimizer = get_optimizer(args, model)
        loss = get_loss(args)
        scheduler = get_lr_scheduler(args, optimizer)

        return cls(
            model,
            train_dataloader,
            optimizer,
            loss,
            args.epochs,
            eval_dataloader,
            scheduler=scheduler,
            **kwargs)

    
    def save_checkpoint(self, epoch, is_best=False):
        checkpoint = {'epoch': epoch}
        for k, v in vars(self).items():
            if k not in self.exclude_from_checkpoint:
                try:
                    checkpoint[k] = v.state_dict()
                except AttributeError:
                    checkpoint[k] = v
        path = os.path.join(self.checkpoint_dir, f"checkpoint_{epoch}{'_best' if is_best else ''}.pt")
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint for epoch {epoch} saved.")


    def train_step(self, batch):
        '''Subclass or change this method with MethodType to customize behavior.'''
        batch = self.process_batch(batch)
        model_input = self.build_model_input(batch)
        self.optimizer.zero_grad()
        model_output = self.model(model_input)
        loss_inputs, loss_targets = self.build_loss_input(batch, model_output)
        # model output, gt
        l = self.loss(*loss_inputs, *loss_targets)
        l.backward()
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        return l


    def process_batch(self, batch):
        n_nodes, coords, gt_tours, gt_lens = batch
        if coords.device != self.device:
            coords = coords.to(self.device)
        if not torch.is_floating_point(coords):
            coords = coords.to(torch.float32)
        return n_nodes, coords, gt_tours, gt_lens

    
    def build_model_input(self, batch):
        return batch[1]

    
    def build_loss_input(self, batch, model_output):
        inputs = (model_output[1],)
        targets = (gt_matrix_from_tour(batch[2][..., :-1] - 1).to(self.device),)
        return inputs, targets

    
    def eval_step(self, batch):
        '''Subclass or change this method with MethodType to customize behavior.'''
        batch = self.process_batch(batch)
        model_input = self.build_model_input(batch)
        model_output = self.model(model_input)
        loss_inputs, loss_targets = self.build_loss_input(batch, model_output)
        l = self.loss(*loss_inputs, *loss_targets)
        if self.metrics:
            metrics_results = {}
            for metric_name, metric_fun in self.metrics.items():
                # model output, gt
                metrics_results[metric_name] = metric_fun(model_output, batch)
        return l, metrics_results

    
    def do_train(self):
        for epoch in range(self.start_epoch, self.epochs):
            self.model.train()
            epoch_loss = 0
            n_samples = 0
            for i, batch in enumerate(tqdm(self.train_dataloader, desc=f"Epoch {epoch}/{self.epochs}")):
                step_loss = self.train_step(batch)
                epoch_loss += step_loss.item()
                n_samples += len(batch)
            
            if n_samples:
                # TODO: log to tensorboard
                epoch_loss /= n_samples
                logger.info(f"[epoch {epoch}] loss: {epoch_loss} | Min loss: {self.best_loss}")
                if epoch_loss < self.best_loss:
                    logger.info(f"[epoch {epoch}] New min loss: {epoch_loss}")
                    self.best_loss = epoch_loss
                    self.save_checkpoint(epoch, True)

            # TODO: run evaluation
            # TODO: save checkpoint
            if epoch and epoch % self.save_epochs == 0:
                self.save_checkpoint(epoch)
        
        self.save_checkpoint(epoch)
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
        d_model_ = d_model
        warm_up = warmup_steps
        s += 1
        return (d_model_ ** -.5) * min(s ** -.5, s * warm_up ** -1.5)
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