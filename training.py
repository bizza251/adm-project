import pathlib
import torch
from dataset import GraphDataset, RandomGraphDataset, gt_matrix_from_tour
from models.custom_transformer import TSPCustomTransformer, TSPTransformer
import logging
from tqdm import tqdm
from torch.optim import Adam, SGD
import torch.nn  as nn
from torch.optim.lr_scheduler import LambdaLR
import os

from models.utility import TourLossReinforce
<<<<<<< HEAD
from torch.utils.tensorboard import SummaryWriter
=======
from utility import BatchGraphInput, custom_collate_fn

>>>>>>> random_data


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(
    format="%(levelname)s: %(message)s"
)



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
        self.best_epoch = -1

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
        if self.checkpoint_dir:
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
        if batch.coords.device != self.device:
            batch.coords = batch.coords.to(self.device)
        if not torch.is_floating_point(batch.coords):
            batch.coords = batch.coords.to(torch.float32)
        return batch

    
    def build_model_input(self, batch):
        return batch.coords

    
    def build_loss_input(self, batch, model_output):
        inputs = (model_output[1],)
        targets = (gt_matrix_from_tour(batch.gt_tour[..., :-1] - 1).to(self.device),)
        return inputs, targets

    
    def eval_step(self, batch):
        '''Subclass or change this method with MethodType to customize behavior.'''
        batch = self.process_batch(batch)
        model_input = self.build_model_input(batch)
        model_output = self.model(model_input)
        loss_inputs, loss_targets = self.build_loss_input(batch, model_output)
        l = self.loss(*loss_inputs, *loss_targets)
        metrics_results = {}
        if self.metrics:
            for metric_name, metric_fun in self.metrics.items():
                # model output, gt
                metrics_results[metric_name] = metric_fun(model_output, batch)
        return l, metrics_results

    
    def do_eval(self):
        self.model.eval()
        logger.info("***** Running evaluation *****")
        eval_loss = 0
        n_samples = 0
        with torch.no_grad():
            for batch in tqdm(self.eval_dataloader, desc="Evaluation...", mininterval=0.5, miniters=2):
                step_loss, metrics_results = self.eval_step(batch)
                eval_loss += step_loss.item()
                if isinstance(batch, (torch.Tensor, BatchGraphInput)):
                    n_samples += len(batch)
                else:
                    n_samples += len(batch[0])
        eval_loss /= n_samples
        logger.info("***** evaluation completed *****")
        logger.info(f"Evaluation loss: {eval_loss}")
        return eval_loss, metrics_results


    def do_train(self):
        writer = SummaryWriter()
        #j=0
        
        for epoch in range(self.start_epoch, self.epochs):
            self.model.train()
            
            epoch_loss = 0
            n_samples = 0
            for batch in tqdm(self.train_dataloader, desc=f"Epoch {epoch}/{self.epochs}", mininterval=1, miniters=5):
                step_loss = self.train_step(batch)
                epoch_loss += step_loss.item()
                if isinstance(batch, (torch.Tensor, BatchGraphInput)):
                    n_samples += len(batch)
                else:
                    n_samples += len(batch[0])
            
            if n_samples:
                # TODO: log to tensorboard
                epoch_loss /= n_samples
                logger.info(f"[epoch {epoch}] Train loss: {epoch_loss}")

            # TODO: run evaluation
<<<<<<< HEAD
            # TODO: save checkpoint
            writer.add_scalar("Loss/train", epoch_loss, epoch)

            if epoch and epoch % self.save_epochs == 0:
=======
            eval_loss, _ = self.do_eval()
            new_best = eval_loss < self.best_loss
            logger.info(f"[epoch {epoch}] Eval loss: {eval_loss} | Min is {self.best_loss} (epoch {self.best_epoch})")
            if new_best:
                    logger.info(f"[epoch {epoch}] New min eval loss: {eval_loss}")
                    self.best_loss = eval_loss
                    self.best_epoch = epoch
                    self.save_checkpoint(epoch, True)

            if not new_best and epoch and epoch % self.save_epochs == 0:
>>>>>>> random_data
                self.save_checkpoint(epoch)
        
        self.save_checkpoint(epoch)
        logger.info("Training completed!")
        writer.close()



class ReinforceTrainer(Trainer):

    def build_loss_input(self, batch, model_output):
        tours, sum_log_probs = model_output
        inputs = (sum_log_probs,)
        coords, gt_len = batch.coords, batch.gt_len
        targets = (coords[torch.arange(len(tours)).view(-1, 1), tours], gt_len.to(sum_log_probs.device))
        return inputs, targets



class CustomReinforceTrainer(Trainer):

    def build_loss_input(self, batch, model_output):
        tours, attn_matrix = model_output
        sum_log_probs = torch.max(attn_matrix, dim=-1)[0].sum(dim=-1)
        inputs = (sum_log_probs,)
        coords, gt_len = batch.coords, batch.gt_len
        targets = (coords[torch.arange(len(tours)).view(-1, 1), tours], gt_len.to(sum_log_probs.device))
        return inputs, targets



def get_model(args):
    if args.model == 'custom':
        model = TSPCustomTransformer.from_args(args)
    elif args.model == 'baseline':
        model = TSPTransformer.from_args(args)
    else:
        raise NotImplementedError()
    return model.to(args.device)


def get_optimizer(args, model):
    if args.optimizer == 'adam':
        return Adam(model.parameters(), lr=args.learning_rate)
    elif args.optimizer == 'sgd':
        return SGD(model.parameters(), lr=args.learning_rate)
    else:
        raise NotImplementedError()


def get_train_dataset(args):
    if args.train_dataset == 'custom':
        return GraphDataset()
    else:
        return RandomGraphDataset(args.train_dataset)


def get_eval_dataset(args):
    if args.train_dataset == 'custom':
        return GraphDataset()
    else:
        return RandomGraphDataset(args.eval_dataset)


def get_train_dataloader(args):
    if args.do_train:
        dataset = get_train_dataset(args)
        return torch.utils.data.DataLoader(
            dataset, 
            batch_size=args.train_batch_size,
            num_workers=1,
            collate_fn=custom_collate_fn)


def get_eval_dataloader(args):
    if args.do_eval:
        dataset = get_eval_dataset(args)
        return torch.utils.data.DataLoader(
            dataset, 
            batch_size=args.eval_batch_size,
            num_workers=1,
            collate_fn=custom_collate_fn)


def get_loss(args):
    if args.loss == 'mse':
        loss = nn.MSELoss()
    elif args.loss == 'reinforce_loss':
        loss = TourLossReinforce()
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


def get_trainer(args):
    if args.train_mode == 'supervised':
        if args.model != 'custom':
            raise NotImplementedError()
        else:
            trainer = Trainer.from_args(args)
    elif args.train_mode == 'reinforce':
        if args.model == 'baseline':
            trainer = ReinforceTrainer.from_args(args)
        elif args.model == 'custom':
            trainer = CustomReinforceTrainer.from_args(args)
    else:
        raise NotImplementedError()
    return trainer


if __name__ == '__main__':
    dataset = GraphDataset()
    loader = torch.utils.data.DataLoader(dataset, batch_size=4)
    model = TSPCustomTransformer(nhead = 4)
    loss = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train(loader, model, loss, optimizer, 50)