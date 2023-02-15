from training.trainer import Trainer
from training.trainer_factory import get_trainer
import argparse



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_eval', action='store_true')
    parser.add_argument('--do_test', action='store_true')
    parser.add_argument('--do_profile', action='store_true')

    # training args
    parser.add_argument('--train_mode', type=str, choices=['supervised', 'reinforce', 'ppo'], default='supervised')
    parser.add_argument('--train_batch_size', type=int, default=16)
    parser.add_argument('--eval_batch_size', type=int, default=16)
    parser.add_argument('--optimizer', type=str, choices=['adam', 'sgd'], default='adam')
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--loss', type=str, choices=['mse', 'reinforce_loss', 'reinforce_loss_mixed', 'custom_reinforce_loss'], default='mse')
    parser.add_argument('--checkpoint_dir', type=str, default=None)
    parser.add_argument('--resume_from_checkpoint', type=str, default=None)
    parser.add_argument('--train_dataset', type=str, default=None)
    parser.add_argument('--eval_dataset', type=str, default=None)
    parser.add_argument('--lr_scheduler', type=str, choices=['transformer'], default=None)
    parser.add_argument('--warmup_steps', type=int, default=None)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--save_epochs', type=int, default=5)
    parser.add_argument('--dataloader_num_workers', type=int, default=0)
    parser.add_argument('--metrics', nargs='*', type=str, default=None)
    parser.add_argument('--tb_comment', type=str, default='')
    parser.add_argument('--reinforce_baseline', type=str, choices=['gt', 'baseline'], default='gt')
    
    # model args
    parser.add_argument('--model', type=str, choices=['custom', 'baseline'], default='custom')
    parser.add_argument('--in_features', type=int, default=2)
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--nhead', type=int, default=4)
    parser.add_argument('--dim_feedforward', type=int, default=512)
    parser.add_argument('--dropout_p', type=float, default=0.)
    parser.add_argument('--activation', type=str, choices=['relu', 'tanh'], default='relu')
    parser.add_argument('--norm', type=str, default='layer')
    parser.add_argument('--norm_eps', type=float, default=1e-5)
    parser.add_argument('--norm_first', type=bool, default=False)
    parser.add_argument('--num_hidden_encoder_layers', type=int, default=2)
    parser.add_argument('--sinkhorn_tau', type=float, default=5e-2)
    parser.add_argument('--sinkhorn_i', type=int, default=20)
    parser.add_argument('--add_cross_attn', type=bool, default=True)
    parser.add_argument('--use_q_proj_ca', type=bool, default=False)
    parser.add_argument('--use_feedforward_block_sa', action='store_true')
    parser.add_argument('--use_feedforward_block_ca', action='store_true')
    parser.add_argument('--positional_encoding', type=str, choices=['sin', 'custom_sin', 'custom'], default='sin')
    # baseline
    parser.add_argument('--num_encoder_layers', type=int, default=3)
    parser.add_argument('--num_hidden_decoder_layers', type=int, default=2)
    parser.add_argument('--clip_logit_c', type=int, default=None)

    # PPO
    # parser.add_argument('--n_envs', type=int, default=8)
    parser.add_argument('--env_nodes', type=int, default=50)
    parser.add_argument('--total_update_steps', type=int, default=int(1e6))
    parser.add_argument('--rollout_steps', type=int, default=128)
    parser.add_argument('--update_batch_size', type=int, default=64)
    parser.add_argument('--update_epochs', type=int, default=10)
    parser.add_argument('--ppo_eps', type=float, default=0.2)
    parser.add_argument('--steps_per_eval', type=int, default=1)

    # ILS
    parser.add_argument('--ils_n_restarts', type=int, default=5)
    parser.add_argument('--ils_n_iterations', type=int, default=10)
    parser.add_argument('--ils_n_permutations', type=int, default=15)
    parser.add_argument('--ils_n_permutations_hillclimbing', type=int, default=7)
    parser.add_argument('--ils_k', type=int, default=0)
    parser.add_argument('--ils_max_perturbs', type=int, default=None)

    # profiling
    parser.add_argument('--filename', type=str, default='')


    args = parser.parse_args()

    trainer = get_trainer(args)
    if args.do_train:
        train_result = trainer.do_train()
    elif args.do_eval:
        eval_result = trainer.do_eval()
    elif args.do_profile:
        profile_result = trainer.do_eval()