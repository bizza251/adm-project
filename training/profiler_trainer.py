from collections import namedtuple
import sys
from tqdm import tqdm
from profiling.utility import get_stats_metrics
from training.trainer import Trainer
import torch 
import numpy as np
import cProfile
import pstats
from uuid import uuid4
from training.utility import get_model, load_checkpoint, logger
import csv
from utility import BatchGraphInput
from dataset import get_dataloader, get_dataset
import os



class StatsTrainer(Trainer):

    def __init__(
            self, 
            model,
            dataset,
            filename,
            resume_from_checkpoint=None,
            device='cpu',
            metrics={},
            *args,
            **kwargs
    ):
        super().__init__(
            model,
            None,
            None,
            None,
            None,
            dataset,
            None,
            None,
            resume_from_checkpoint,
            device,
            None,
            None,
            None,
            *args,
            **kwargs
        )

        filename = filename if filename.endswith('.csv') else filename + '.csv'
        if not filename.startswith('/'):
            os.system("mkdir -p profiling_results")
            filename = 'profiling_results/' + filename
        self.filename = filename
        self.metrics = metrics
        self.skip_metrics = {'total_time', 'gt_len', 'id'}
        for name in self.skip_metrics:
            self.metrics[name] = []
        
        self.row = namedtuple('row', self.metrics.keys())
        self.f = open(self.filename, "w")
        self.csv_writer = csv.DictWriter(self.f, fieldnames=self.row._fields)
        self.csv_writer.writeheader()

    
    def __del__(self):
        self.f.close()
        

    def eval_step(self, batch):
        batch = self.process_batch(batch)
        model_input = self.build_model_input(batch)
        profiler = cProfile.Profile()
        profiler.enable()
        model_output = self.model(*model_input)
        profiler.disable()
        stats = pstats.Stats(profiler, stream=sys.stdout).sort_stats('cumtime')
        metrics_results = {}
        for metric_name, metric_fun in self.metrics.items():
            if metric_name not in self.skip_metrics:
                metrics_results[metric_name] = metric_fun(model_output, batch)
        metrics_results['total_time'] = stats.total_tt
        metrics_results['gt_len'] = batch.gt_len.cpu().numpy()
        metrics_results['id'] = batch.id
        return metrics_results
    

    def do_eval(self):
        metrics_results = {}
        self.model.eval()
        logger.info("***** Running evaluation *****")
        n_samples = 0
        metrics_results = {k: [] for k in self.metrics.keys()}
        with torch.no_grad():
            for batch in tqdm(self.eval_dataloader, desc="Evaluation...", mininterval=0.5, miniters=2):
                step_metrics_results = self.eval_step(batch)
                flattened_metrics = {k: [] for k in step_metrics_results.keys()}
                for metric_name, metric_value in step_metrics_results.items():
                    flattened_metrics[metric_name].extend(np.array(metric_value).reshape(-1).tolist())
                if len(batch) > 1:
                    flattened_metrics['total_time'].extend([np.nan for _ in range(1, len(batch))])
                for i in range(len(batch)):
                    row = self.row(**{k: v[i] for k, v in flattened_metrics.items()})
                    self.csv_writer.writerow(row._asdict())
        
                if isinstance(batch, (torch.Tensor, BatchGraphInput)):
                    n_samples += len(batch)
                else:
                    n_samples += len(batch[0])
        logger.info("***** evaluation completed *****")
        logger.info(f"Processed sample: {n_samples}")
        return metrics_results

    
    @classmethod
    def from_args(cls, args):
        model = get_model(args)
        checkpoint = load_checkpoint(args.resume_from_checkpoint, verbose=False)
        model.load_state_dict(checkpoint['model'])
        eval_dataset = get_dataset(args.eval_dataset) 
        metrics = get_stats_metrics(args)
        if args.filename:
            filename = args.filename
        else:
            filename = args.model + '_' + str(uuid4())
        return cls(model, eval_dataset, filename, None, args.device, metrics, None, None, 
                   eval_batch_size=args.eval_batch_size, dataloader_num_workers=args.dataloader_num_workers)
