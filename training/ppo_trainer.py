from tqdm import tqdm
from environments.env import get_env
from ppo.agent import PPOAgent
from ppo.ppo import PPO
from training.trainer import CustomReinforceTrainer, Trainer, ReinforceTrainer
from training.utility import get_training_commons
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from utility import logger



class PPOTrainer(ReinforceTrainer):

    exclude_from_checkpoint = {
        *Trainer.exclude_from_checkpoint,
        'ppo',
    }

    def __init__(
        self,
        model,
        train_dataset,
        optimizer,
        loss,
        env,
        steps,
        rollout_steps,
        update_epochs,
        update_batch_size,
        ppo_eps,
        eval_dataset=None,
        scheduler=None,
        checkpoint_dir=None,
        resume_from_checkpoint=None,
        device='cpu',
        metrics=None,
        save_epochs=5,
        tb_comment='',
        steps_per_eval=1,
        *args,
        **kwargs
    ):
        ppo_agent = PPOAgent(model)
        super().__init__(
            ppo_agent, 
            train_dataset, 
            optimizer, 
            loss, 
            -1, 
            eval_dataset, 
            scheduler, 
            checkpoint_dir, 
            resume_from_checkpoint,
            device, 
            metrics, 
            save_epochs, 
            tb_comment, 
            *args, 
            **kwargs)
        
        self.train_dataloader = None

        self.ppo = PPO(
            ppo_agent,
            env,
            optimizer,
            steps,
            rollout_steps,
            update_epochs,
            update_batch_size,
            ppo_eps
        )

        self.steps_per_eval = steps_per_eval


    @classmethod
    def from_args(cls, args):
        training_commons = get_training_commons(args)
        kwargs = {k:v for k, v in training_commons.kwargs.items() if k not in ('steps', 'rollout_steps', 'update_epochs', 'update_batch_size', 'ppo_eps')}
        env = get_env(args, training_commons.train_dataset)

        samples_per_rollout = args.rollout_steps * args.train_batch_size
        updates_per_step = samples_per_rollout / args.update_batch_size * args.update_epochs
        total_update_steps = int(np.ceil(args.total_update_steps / updates_per_step))

        return cls(
            training_commons.model,
            training_commons.train_dataset,
            training_commons.optimizer,
            training_commons.loss,
            env,
            total_update_steps,
            args.rollout_steps,
            args.update_epochs,
            args.update_batch_size,
            args.ppo_eps,
            training_commons.eval_dataset,
            training_commons.scheduler,
            metrics=training_commons.metrics,
            **kwargs,
        )


    def do_train(self):
        writer = SummaryWriter(comment=self.tb_comment)

        for step in tqdm(range(self.ppo.steps)):
            self.ppo.train_step()

            logger.info("PPO step {step} completed.")
            
            if step and step % self.steps_per_eval == 0:
                eval_loss, metrics_results = self.do_eval()
                writer.add_scalar("Loss/eval", eval_loss, step)
                new_best = eval_loss < self.best_loss
                logger.info(f"[step {step}] Eval loss: {eval_loss} | Min is {self.best_loss} (step {self.best_epoch})")
                if new_best:
                        logger.info(f"[step {step}] New min eval loss: {eval_loss}")
                        self.best_loss = eval_loss
                        self.best_epoch = step
                        self.save_checkpoint(step, True)

                self.update_metrics(metrics_results)
                for k, v in metrics_results.items():
                    writer.add_scalar(f"Metrics/{k}", v, step)

                if not new_best and step and step % self.save_epochs == 0:
                    self.save_checkpoint(step)
        
        self.save_checkpoint(step)
        logger.info("Training completed!")
        writer.close()