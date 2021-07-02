import argparse
import numpy as np


def parse_args():
	parser = argparse.ArgumentParser()

	# environment
	parser.add_argument('--domain_name', default='walker')
	parser.add_argument('--task_name', default='walk')
	parser.add_argument('--frame_stack', default=3, type=int)
	parser.add_argument('--action_repeat', default=4, type=int)
	parser.add_argument('--episode_length', default=1000, type=int)
	parser.add_argument('--train_mode', default='original', type=str)
	parser.add_argument('--eval_mode', default='color_hard', type=str)
	
	# agent
	parser.add_argument('--algorithm', default='svea', type=str)
	parser.add_argument('--train_steps', default='500k', type=str)
	parser.add_argument('--discount', default=0.99, type=float)
	parser.add_argument('--init_steps', default=1000, type=int)
	parser.add_argument('--batch_size', default=512, type=int)
	parser.add_argument('--hidden_dim', default=1024, type=int)
	parser.add_argument('--crop', default='vitcrop', type=str)

	# actor
	parser.add_argument('--actor_lr', default=1e-3, type=float)
	parser.add_argument('--actor_beta', default=0.9, type=float)
	parser.add_argument('--actor_log_std_min', default=-10, type=float)
	parser.add_argument('--actor_log_std_max', default=2, type=float)
	parser.add_argument('--actor_update_freq', default=2, type=int)

	# critic
	parser.add_argument('--critic_lr', default=1e-3, type=float)
	parser.add_argument('--critic_beta', default=0.9, type=float)
	parser.add_argument('--critic_tau', default=0.01, type=float)
	parser.add_argument('--critic_target_update_freq', default=2, type=int)

	# architecture
	parser.add_argument('--num_shared_layers', default=11, type=int)
	parser.add_argument('--num_head_layers', default=0, type=int)
	parser.add_argument('--num_filters', default=32, type=int)
	parser.add_argument('--projection_dim', default=100, type=int)
	parser.add_argument('--encoder_tau', default=0.05, type=float)

	# entropy maximization
	parser.add_argument('--init_temperature', default=0.1, type=float)
	parser.add_argument('--alpha_lr', default=1e-4, type=float)
	parser.add_argument('--alpha_beta', default=0.5, type=float)

	# svea w/ vit
	parser.add_argument('--svea_aug', default='conv', type=str)
	parser.add_argument('--patch_size', default=8, type=int)
	parser.add_argument('--embed_dim', default=128, type=int)
	parser.add_argument('--depth', default=4, type=int)
	parser.add_argument('--num_heads', default=8, type=int)
	parser.add_argument('--mlp_ratio', default=1., type=float)
	parser.add_argument('--qvk_bias', default=False, action='store_true')
	parser.add_argument('--svea_weight_decay', default=False, action='store_true')

	# eval
	parser.add_argument('--save_freq', default='500k', type=str)
	parser.add_argument('--eval_freq', default='20k', type=str)
	parser.add_argument('--eval_episodes', default=10, type=int)

	# misc
	parser.add_argument('--seed', default=1, type=int)
	parser.add_argument('--exp_suffix', default='default', type=str)
	parser.add_argument('--log_dir', default='logs', type=str)
	parser.add_argument('--save_video', default=False, action='store_true')

	args = parser.parse_args()

	assert args.algorithm in {'svea'}, f'specified algorithm "{args.algorithm}" is not supported'
	assert args.crop in {'vitcrop'}, f'specified crop "{args.crop}" is not supported'
	assert args.svea_aug in {'conv', 'none'}, f'specified augmentation "{args.svea_aug}" is not supported'

	modes = {'original', 'color_easy', 'color_hard', 'video_easy', 'video_hard'}
	assert args.train_mode in modes and args.eval_mode in modes, f'specified mode "{args.train_mode}" or "{args.eval_mode}" is not supported'
	assert args.seed is not None, 'must provide seed for experiment'
	assert args.exp_suffix is not None, 'must provide an experiment suffix for experiment'
	assert args.log_dir is not None, 'must provide a log directory for experiment'

	args.train_steps = int(args.train_steps.replace('k', '000'))
	args.save_freq = int(args.save_freq.replace('k', '000'))
	args.eval_freq = int(args.eval_freq.replace('k', '000'))
	
	return args
