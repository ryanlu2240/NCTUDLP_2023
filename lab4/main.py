import os
import pdb
import random
import argparse
import numpy as np

import torch

## Self-defined
from dataloader import create_dataset
from model.build_model import build_model
from trainer import build_trainer
from util import save_exp, save_learning_curves, plot_confusion_matrix

def parse_args():
	parser = argparse.ArgumentParser()

	## Mode
	parser.add_argument("--train", action="store_true")
	parser.add_argument("--test" , action="store_true", help="Load previous checkpoint for testing only.")

	## Training options
	parser.add_argument("--train_epoch", type=int, default=20)
	parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
	parser.add_argument("--bs", type=int, default=4, help="Batch size during training")
	parser.add_argument("--optim", type=str, default="sgd")
	parser.add_argument("--momentum", type=float, default=0.9)
	parser.add_argument("--weight_decay", type=float, default=5e-4)
	parser.add_argument("--beta1", type=float, default=0.9)
	parser.add_argument("--beta2", type=float, default=0.999)
	parser.add_argument("--model", type=str, default="resnet18", choices=["resnet18", "resnet50"])
	parser.add_argument("--pretrained", action="store_true", help="Use pretrained models or not.")
	parser.add_argument("--loss_weight", action="store_true", help="Use weighted loss for training or not.")
	parser.add_argument("--report_every", type=int, default=1, help="Display results every 1 epochs.")
	parser.add_argument("--seed", type=int, default=None, help="Whether to fix random seed or not.")
	parser.add_argument("--device", type=str, default="cuda")

	## Save what
	parser.add_argument("--save_exp", action="store_true", help="Whether to save the training result or not.")
	parser.add_argument("--save_confusion_matrix", action="store_true", help="Only available when test only.")

	## Paths
	parser.add_argument("--root_path", type=str, default="./")
	parser.add_argument("--result_path", type=str, default="./result")
	parser.add_argument("--checkpoint_path", type=str, default="./checkpoints")

	args = parser.parse_args()

	return args

if __name__ == "__main__":
	args = parse_args()

	## Set device
	device = torch.device(args.device)
	print("\nDevice: {}\n".format(device))

	if device.type != "cuda":
		raise SystemExit("Not using GPU for training!")

	## Set random seed
	if args.seed is not None:
		print("Random seed: {}\n".format(args.seed))
		torch.manual_seed(args.seed)
		torch.cuda.manual_seed(args.seed)
		torch.cuda.manual_seed_all(args.seed)
		np.random.seed(args.seed)
		random.seed(args.seed)
		torch.backends.cudnn.benchmark = False
		torch.backends.cudnn.deterministic = True

	## Build dataset
	num_class, loss_weight, train_loader, test_loader = create_dataset(args)
	#loss_weight = [1, 10, 5, 10, 10]

	## Build model
	model = build_model(args, num_class)

	## Build trainer & train (or test)
	trainer = build_trainer(args, model, device)
	if args.train:
		print("Start training...")
		best, epochs, train_accs, test_accs = trainer.train(train_loader, test_loader, loss_weight)
		
		## Save experiment results
		if args.save_exp:
			save_exp(args, best)
			save_learning_curves(args, epochs, train_accs, test_accs)

	elif args.test:
		print("Test only...")
		test_acc, test_labels, test_preds = trainer.test(test_loader)

		print(
			(f"Test Accuracy: {test_acc:.4f}\n"
			)
		)

		if args.save_confusion_matrix:
			plot_confusion_matrix(args, test_labels, test_preds)


