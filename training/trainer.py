import pathlib
import numpy as np
import torch
from dataset import get_dataloader, get_dataset, gt_matrix_from_tour
from models.custom_transformer import TSPCustomTransformer, TSPTransformer
from tqdm import tqdm
import os
from torch.utils.tensorboard import SummaryWriter
from utility import BatchGraphInput, get_tour_coords, get_tour_len, iterated_local_search, path_cost
from training.utility import *


class Trainer:

    exclude_from_checkpoint = {
        'train_dataset',
        'eval_dataset',
        'eval_dataloader',
        'save_epochs',
        'epochs',
        'eval_set',
        'checkpoint_dir',
        'metrics',
        'loss',
        'device'
    }

    def __init__(
        self,
        model,
        train_dataset,
        optimizer,
        loss,
        epochs,
        eval_dataset=None,
        scheduler=None,
        checkpoint_dir=None,
        resume_from_checkpoint=None,
        device='cpu',
        metrics=None,
        save_epochs=5,
        tb_comment='',
        *args,
        **kwargs
    ):

        self.model = model
        self.train_dataloader = get_dataloader(train_dataset, kwargs.get('train_batch_size'), kwargs.get('dataloader_num_workers'))
        self.optimizer = optimizer
        self.loss = loss
        self.epochs = epochs
        self.eval_dataloader = get_dataloader(eval_dataset, kwargs.get('eval_batch_size'), kwargs.get('dataloader_num_workers'))
        self.scheduler = scheduler
        self.checkpoint_dir = checkpoint_dir
        self.device = device
        self.metrics = metrics
        self.save_epochs = save_epochs
        self.tb_comment = tb_comment 

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
        if self.loss is not None:
            self.loss.to(device)


    @classmethod
    def from_args(cls, args):
        training_commons = get_training_commons(args)

        return cls(
            training_commons.model,
            training_commons.train_dataset,
            training_commons.optimizer,
            training_commons.loss,
            args.epochs,
            training_commons.eval_dataset,
            scheduler=training_commons.scheduler,
            metrics=training_commons.metrics,
            **training_commons.kwargs)
    
    
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


    def update_metrics(self, metrics_results):
        for k, v in metrics_results.items():
            if k not in self.best_metrics:
                self.best_metrics[k] = v
            else:
                if v < self.best_metrics[k]:
                    # TODO: currently we assume `better` means `less than`.
                    logger.info(f"New best for metric {k}: {v} (previous was {self.best_metrics[k]}).")
                    self.best_metrics[k] = v


    def train_step(self, batch):
        '''Subclass or change this method with MethodType to customize behavior.'''
        batch = self.process_batch(batch)
        model_input = self.build_model_input(batch)
        model_output = self.model(*model_input)
        loss_inputs, loss_targets = self.build_loss_forward_input(batch, model_output)
        # model output, gt
        l = self.loss(*loss_inputs, *loss_targets)
        self.optimizer.zero_grad()
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
        return (batch.coords, )

    
    def build_loss_inputs(self, batch, model_output):
        return (model_output.attn_matrix,)

    
    def build_loss_targets(self, batch, model_output):
        return (gt_matrix_from_tour(batch.gt_tour[..., :-1] - 1).to(self.device),)


    def build_loss_forward_input(self, batch, model_output):
        inputs = self.build_loss_inputs(batch, model_output)
        targets = self.build_loss_targets(batch, model_output)
        return inputs, targets

    
    def eval_step(self, batch):
        '''Subclass or change this method with MethodType to customize behavior.'''
        batch = self.process_batch(batch)
        model_input = self.build_model_input(batch)
        model_output = self.model(*model_input)
        loss_inputs, loss_targets = self.build_loss_forward_input(batch, model_output)
        l = self.loss(*loss_inputs, *loss_targets)
        metrics_results = {}
        if self.metrics:
            for metric_name, metric_fun in self.metrics.items():
                # model output, gt
                metrics_results[metric_name] = metric_fun(model_output, batch)
        return l, metrics_results

    
    def do_eval(self):
        eval_loss, metrics_results = torch.inf, {}
        if self.eval_dataloader is not None:
            self.model.eval()
            logger.info("***** Running evaluation *****")
            eval_loss = 0
            n_samples = 0
            n_batches = 0
            metrics_results = {k: [] for k in self.metrics.keys()}
            with torch.no_grad():
                for batch in tqdm(self.eval_dataloader, desc="Evaluation...", mininterval=0.5, miniters=2):
                    step_loss, step_metrics_results = self.eval_step(batch)
                    eval_loss += step_loss.item()
                    for metric_name, metric_value in step_metrics_results.items():
                        metrics_results[metric_name].append(metric_value)
                    if isinstance(batch, (torch.Tensor, BatchGraphInput)):
                        n_samples += len(batch)
                    else:
                        n_samples += len(batch[0])
                    n_batches += 1
            eval_loss /= n_batches
            logger.info("***** evaluation completed *****")
            logger.info(f"Eval loss: {eval_loss} | Processed sample: {n_samples}")
            for metric_name in metrics_results.keys():
                avg = np.mean(metrics_results[metric_name])
                metrics_results[metric_name] = avg
                logger.info(f"Eval `{metric_name}`: {metrics_results[metric_name]}")
        return eval_loss, metrics_results


    def epoch_begin_hook(self):
        self.model.train()


    def do_train(self):
        writer = SummaryWriter(comment=self.tb_comment)
        #j=0
        
        for epoch in range(self.start_epoch, self.epochs):
            self.epoch_begin_hook()
            
            epoch_loss = 0
            n_samples = 0
            n_batches = 0
            for i, batch in enumerate(tqdm(self.train_dataloader, desc=f"Epoch {epoch}/{self.epochs}", mininterval=1, miniters=5)):
                step_loss = self.train_step(batch)
                epoch_loss += step_loss.item()
                if isinstance(batch, (torch.Tensor, BatchGraphInput)):
                    n_samples += len(batch)
                else:
                    n_samples += len(batch[0])
                if i % 5 == 0:
                    writer.add_scalar("train/learning rate", self.optimizer.param_groups[0]['lr'], (i + 1) * (epoch + 1))
                n_batches += 1
            
            if n_samples:
                # TODO: log to tensorboard
                epoch_loss /= n_batches
                logger.info(f"[epoch {epoch}] Train loss: {epoch_loss} | Processed sample: {n_samples}")

            writer.add_scalar("Loss/train", epoch_loss, epoch)

            eval_loss, metrics_results = self.do_eval()
            writer.add_scalar("Loss/eval", eval_loss, epoch)
            new_best = eval_loss < self.best_loss
            logger.info(f"[epoch {epoch}] Eval loss: {eval_loss} | Min is {self.best_loss} (epoch {self.best_epoch})")
            if new_best:
                    logger.info(f"[epoch {epoch}] New min eval loss: {eval_loss}")
                    self.best_loss = eval_loss
                    self.best_epoch = epoch
                    self.save_checkpoint(epoch, True)
            
            self.update_metrics(metrics_results)
            for k, v in metrics_results.items():
                writer.add_scalar(f"Metrics/{k}", v, epoch)

            if not new_best and epoch and epoch % self.save_epochs == 0:
                self.save_checkpoint(epoch)
        
        self.save_checkpoint(epoch)
        logger.info("Training completed!")
        writer.close()



class ReinforceTrainer(Trainer):

    def build_loss_inputs(self, batch, model_output):
        return (batch.coords, model_output.sum_log_probs, model_output.tour)

    
    def build_loss_targets(self, batch, model_output):
        return (batch.gt_len.to(model_output.sum_log_probs.device), batch.gt_tour, model_output.attn_matrix)



class BaselineReinforceTrainer(ReinforceTrainer):

    def epoch_begin_hook(self):
        self.model.update_bsln()
        self.model.train()

    
    def build_loss_targets(self, batch, model_output):
        if self.model.training:
            return (get_tour_len(get_tour_coords(batch.coords, model_output.bsln.tour)),)
        else:
            return (batch.gt_len.to(model_output.sum_log_probs.device),)




class CustomReinforceTrainer(ReinforceTrainer):
    
    # def build_loss_inputs(self, batch, model_output):
    #     return (model_output.sum_log_probs, get_tour_coords(batch.coords, model_output.tour), model_output.tour, batch.gt_tour)


    # def build_loss_targets(self, batch, model_output):
    #     return (batch.gt_len.to(model_output.sum_log_probs.device), model_output.attn_matrix, batch.coords)

    def build_model_input(self, batch):
        bsz, nodes = batch.coords.shape[:-1]
        attn_mask = torch.zeros((bsz, nodes, nodes), device=batch.coords.device)
        attn_mask[torch.arange(bsz).view(-1, 1, 1), 0, (batch.gt_tour[:, 0:1].unsqueeze(1) - 1).to(torch.long)] = -1e9
        return (batch.coords, attn_mask)



class CustomBaselineReinforceTrainer(CustomReinforceTrainer, BaselineReinforceTrainer):
    ...



class TestReinforceTrainer(CustomReinforceTrainer):

    def eval_step(self, batch):
        '''Subclass or change this method with MethodType to customize behavior.'''
        batch = self.process_batch(batch)
        model_input = self.build_model_input(batch)
        model_output = self.model(*model_input)

        # create graph objects
        # graphs = []
        # from graph import MyGraph
        # for g in batch.coords:
        #     graphs.append(MyGraph(coords=g))

        # iterated_local_search(path_cost, graphs[10], 50, 100, model_output.tour[10] + 1)
        loss_inputs, loss_targets = self.build_loss_forward_input(batch, model_output)
        l = self.loss(*loss_inputs, *loss_targets)
        metrics_results = {}
        if self.metrics:
            for metric_name, metric_fun in self.metrics.items():
                # model output, gt
                metrics_results[metric_name] = metric_fun(model_output, batch)
        return l, metrics_results
