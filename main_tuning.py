from training.trainer import Trainer
from training.trainer_factory import get_trainer
import argparse
import argparse
import optuna
import logging
import sys
from datetime import datetime
import json


def objective(trial):
    parser = argparse.ArgumentParser()

    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_eval', action='store_true')
    parser.add_argument('--do_test', action='store_true')

    # training args
    parser.add_argument('--train_mode', type=str, choices=['supervised', 'reinforce', 'ppo'], default='supervised')
    parser.add_argument('--train_batch_size', type=int, default=16)
    parser.add_argument('--eval_batch_size', type=int, default=16)
    parser.add_argument('--optimizer', type=str, choices=['adam', 'sgd'], default='adam')
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--loss', type=str, choices=['mse', 'reinforce_loss', 'reinforce_loss_mixed'], default='mse')
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
    parser.add_argument('--positional_encoding', type=str, choices=['sin', 'custom_sin', 'custom'], default='custom_sin')
    #parser.add_argument('--patience', type=int, default=10)
    #parser.add_argument('--ratio_loss_gain', type=float, default=1.005)
    # baseline
    parser.add_argument('--num_encoder_layers', type=int, default=3)
    parser.add_argument('--num_hidden_decoder_layers', type=int, default=2)
    parser.add_argument('--clip_logit_c', type=int, default=None)
    
    
    
    #parser.add_argument('--target_p', type=int, default=50) #probabilità di vedere la target_baseline
    #parser.add_argument('--percentage_improvement_loss', type=float, default=1.00005) #probabilità di vedere la target_baseline
    #parser.add_argument('--patience_probability', type=int, default=1000) #probabilità di vedere la target_baseline
    #parser.add_argument('--target_p_step', type=int, default=1) #probabilità di vedere la target_baseline



    args = parser.parse_args()

    #parameters to be optimized
    args.learning_rate = trial.suggest_float('learning_rate', 1e-7, 1e-2)
    args.train_batch_size = trial.suggest_categorical('train_batch_size', [16, 32, 64, 128, 256, 512])
    args.eval_batch_size = trial.suggest_categorical('eval_batch_size', [16, 32, 64, 128, 256, 512])
    args.num_hidden_encoder_layer = trial.suggest_int('num_hidden_encoder_layer', 2, 6)
    args.dim_feedforward = trial.suggest_categorical('dim_feedforward', [16, 32, 64, 128, 256, 512])
    args.d_model = trial.suggest_categorical('d_model', [128, 256, 512, 1024])
    args.clip_logit_c = trial.suggest_int('clip_logit_c', 5, 20)
    args.positional_encoding = trial.suggest_categorical('positional_encoding', ['sin', 'custom_sin', 'custom'])
    #args.target_p = trial.suggest_int('target_p', 0, 100)
    #args.target_p_step = trial.suggest_int('target_p_step', 0, 5)
    #args.patience_probability = trial.suggest_int('patience_probability', 10e2, 10e5)
    #args.percentage_improvement_loss = trial.suggest_float('percentage_improvement_loss', 1., 1.05)

    ff_usage = trial.suggest_categorical('ff_usage', [
        {
            'block_ca':True,
            'block_sa':True,
        },
        {
            'block_ca':False,
            'block_sa':True,
        },
        {
            'block_ca':True,
            'block_sa':False,
        },
    ])
    
    args.use_feedforward_block_ca = ff_usage['block_ca']
    args.use_feedforward_block_sa = ff_usage['block_sa']
    
    print(type(args.train_batch_size))

    trainer = get_trainer(args)
    if args.do_train:
        train_result = trainer.do_train()
    elif args.do_eval:
        eval_result = trainer.do_eval()
    now = datetime.now()
    run_name = f"run_{now.year}_{now.month}_{now.day}_{now.hour}_{now.minute}"  # Unique identifier of the study.
    params = {}
    for key in trial.params.keys():
        params[key] = trial.params[key]
    params['len_to_gt_len_ratio'] = trainer.best_metrics['len_to_gt_len_ratio']
    params = json.dumps(params, indent = 4)

    with open(f"tuning/params_{study_name}.json", "w") as outfile:
        outfile.write(params)
    return trainer.best_metrics['len_to_gt_len_ratio'] #metric to be optimized
        


if __name__ == '__main__':
    n_trials = 60

    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    now = datetime.now()
    study_name = f"tuning_{now.year}_{now.month}_{now.day}_{now.hour}_{now.minute}"  # Unique identifier of the study.
    storage_name = "sqlite:///{}.db".format(study_name)

    print('To see params dashboard run the following command:')
    print(f'optuna-dashboard {storage_name}')

    study = optuna.create_study(study_name=study_name, storage=storage_name)
    study.optimize(objective, n_trials=n_trials)
    
    print(study.best_params)
    best_hyperparams = json.dumps(study.best_params, indent = 4)
    with open(f"tuning/best_hyperparams_{study_name}.json", "w") as outfile:
        outfile.write(best_hyperparams)
    print('To see params dashboard run the following command:')
    print(f'optuna-dashboard {storage_name}')