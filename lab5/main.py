import os
import random
import argparse
import itertools
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

## Self-defined
from loader import load_train_data, load_test_data
from models.build_models import build_models
from trainer import build_trainer

torch.backends.cudnn.benchmark = True

def parse_args():
	parser = argparse.ArgumentParser()

	## What to do
	parser.add_argument("--train", default=False, action="store_true")
	parser.add_argument("--test" , default=False, action="store_true")

	## Hyper-parameters
	parser.add_argument("--cuda"      , default=False , action="store_true")
	parser.add_argument("--seed"      , default=1     , type=int,    help="manual seed")
	parser.add_argument("--lr"        , default=0.002 , type=float,  help="learning rate")
	parser.add_argument("--beta1"     , default=0.9   , type=float,  help="momentum term for adam")
	parser.add_argument("--batch_size", default=12    , type=int,    help="batch size")
	parser.add_argument("--optimizer" , default="adam",              help="optimizer to train with")
	parser.add_argument("--niter"     , type=int      , default=300, help="number of epochs to train for")
	parser.add_argument("--epoch_size", type=int      , default=600, help="epoch size")

	## Training strategies
	parser.add_argument("--tfr",                   type=float, default=1.0, help="teacher forcing ratio (0 ~ 1)")
	parser.add_argument("--tfr_start_decay_epoch", type=int  , default=0  , help="The epoch that teacher forcing ratio become decreasing")
	parser.add_argument("--tfr_decay_step"       , type=float, default=0  , help="The decay step size of teacher forcing ratio (0 ~ 1)")
	parser.add_argument("--tfr_lower_bound"      , type=float, default=0  , help="The lower bound of teacher forcing ratio for scheduling teacher forcing ratio (0 ~ 1)")
	parser.add_argument("--kl_anneal_cyclical", default=False, action="store_true", help="use cyclical mode")
	parser.add_argument("--kl_anneal_ratio"   , type=float   , default=2          , help="The decay ratio of kl annealing")
	parser.add_argument("--kl_anneal_cycle"   , type=int     , default=3          , help="The number of cycle for kl annealing (if use cyclical mode)")
	parser.add_argument("--n_past"  , type=int, default=2  , help="number of frames to condition on")
	parser.add_argument("--n_future", type=int, default=10 , help="number of frames to predict")
	parser.add_argument("--n_eval"  , type=int, default=30 , help="number of frames to predict at eval time")
	parser.add_argument("--rnn_size", type=int, default=256, help="dimensionality of hidden layer")
	parser.add_argument("--posterior_rnn_layers", type=int, default=1, help="number of layers")
	parser.add_argument("--predictor_rnn_layers", type=int, default=2, help="number of layers")
	parser.add_argument("--z_dim"   , type=int  , default=64    , help="dimensionality of z_t")
	parser.add_argument("--g_dim"   , type=int  , default=128   , help="dimensionality of encoder output vector and decoder input vector")
	parser.add_argument("--beta"    , type=float, default=0, help="weighting on KL to prior")
	parser.add_argument("--cond_dim", type=int  , default=7, help="dimensionality of condition")

	parser.add_argument("--num_workers"    , type=int, default=4, help="number of data loading threads")
	parser.add_argument("--last_frame_skip", action="store_true", help="if true, skip connections go between frame t and frame t+t rather than last ground truth frame")

	## Paths
	parser.add_argument("--log_dir"  , default="./logs/fp", help="base directory to save logs")
	parser.add_argument("--model_dir", default="" , help="base directory to save checkpoints")
	parser.add_argument("--data_root", default="./data", help="root directory for data")
	parser.add_argument("--test_set" , default="test"      , help="validate/test when test only")

	parser.add_argument("--exp_name", default=None, type=str, help="experiment directory name for saving results")

	args = parser.parse_args()
	return args

def main():
	args = parse_args()

	## Set mode
	if args.train:
		mode = "train"
	elif args.test:
		mode = "test"
		test_set = args.test_set

	## Set device
	if args.cuda:
		assert torch.cuda.is_available(), "CUDA is not available."
		device = "cuda"
	else:
		device = "cpu"
	print("\nDevice: {}".format(device))
	
	assert args.n_past + args.n_future <= 30 and args.n_eval <= 30
	assert 0 <= args.tfr and args.tfr <= 1
	assert 0 <= args.tfr_start_decay_epoch 
	assert 0 <= args.tfr_decay_step and args.tfr_decay_step <= 1

	if args.model_dir != "":
		## Load model and continue training from checkpoint
		saved_model = torch.load(f"{args.model_dir}/model.pth")
		optimizer = args.optimizer
		model_dir = args.model_dir
		niter = args.niter
		args = saved_model["args"]
		args.optimizer = optimizer
		args.model_dir = model_dir
		if mode == "train":
			args.log_dir = f"{args.log_dir}/continued"
		elif mode == "test":
			args.log_dir = f"{args.log_dir}/test"
		start_epoch = saved_model["last_epoch"]
	else:
		saved_model = None
		if args.exp_name is None:
			name = f"rnn_size={args.rnn_size}-predictor-posterior-rnn_layers={args.predictor_rnn_layers}-{args.posterior_rnn_layers}-n_past={args.n_past}-n_future={args.n_future}-lr={args.lr}-g_dim={args.g_dim}-z_dim={args.z_dim}-last_frame_skip={args.last_frame_skip}-beta={args.beta}"
		else:
			name = args.exp_name

		args.log_dir = f"{args.log_dir}/{name}"
		niter = args.niter
		start_epoch = 0

	os.makedirs(args.log_dir, exist_ok=True)
	os.makedirs(f"{args.log_dir}/gen/", exist_ok=True)

	## Set seed
	print(f"Random Seed: {args.seed}")
	random.seed(args.seed)
	torch.manual_seed(args.seed)
	torch.cuda.manual_seed_all(args.seed)

	if args.model_dir == "" and os.path.exists(f"{args.log_dir}/train_record.txt"):
		os.remove(f"{args.log_dir}/train_record.txt")
	
	print(f"\nArguments:\n{args}")

	with open(f"{args.log_dir}/train_record.txt", "a") as train_record:
		train_record.write(f"args: {args}\n")

	##################
	## Build Models ##
	################## 
	frame_predictor, posterior, encoder, decoder = build_models(args, saved_model, device)

	##################
	## Load Dataset ##
	##################
	if mode == "train":
		train_data, train_loader, train_iterator, \
		valid_data, valid_loader, valid_iterator = load_train_data(args)
	elif mode == "test":
		test_data, test_loader, test_iterator = load_test_data(args)

	###################
	## Build Trainer ##
	###################
	trainer = build_trainer(args, frame_predictor, posterior, encoder, decoder, device)
	if mode == "train":
		trainer.train(
			start_epoch, niter, 
			train_data, train_loader, train_iterator, 
			valid_data, valid_loader, valid_iterator
		)
	elif mode == "test":
		assert args.model_dir != "", "model_dir should not be empty!"
		trainer.test(test_data, test_loader, test_iterator, test_set)

if __name__ == '__main__':
	main()
		
