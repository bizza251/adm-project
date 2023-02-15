from training.ppo_trainer import PPOTrainer
from training.profiler_trainer import StatsTrainer
from training.trainer import *
from training.utility import *



def get_trainer(args):
    if args.do_profile:
        return StatsTrainer.from_args(args)
    elif args.train_mode == 'supervised':
        if args.model != 'custom':
            raise NotImplementedError()
        else:
            return Trainer.from_args(args)
    elif args.train_mode == 'reinforce':
        if args.model == 'baseline':
            if args.reinforce_baseline == 'baseline':
                return BaselineReinforceTrainer.from_args(args)
            else:
                return ReinforceTrainer.from_args(args)
        elif args.model == 'custom':
            if args.reinforce_baseline == 'baseline':
                return CustomBaselineReinforceTrainer.from_args(args)
            else:
                if not args.do_train:
                    return TestReinforceTrainer.from_args(args)
                else:
                    return CustomReinforceTrainer.from_args(args)
    elif args.train_mode == 'ppo':
        return PPOTrainer.from_args(args)
    else:
        raise NotImplementedError()
