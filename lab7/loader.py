from torch.utils.data import DataLoader

## Self-defined
from dataset import iclevr_dataset

def load_train_data(args):
	print("\nLoading datasets...")

	train_data = iclevr_dataset(args, "train")
	# valid_data = iclevr_dataset(args, "validate")

	train_loader = DataLoader(
		train_data,
		num_workers=args.num_workers,
		batch_size=args.batch_size,
		shuffle=True,
		drop_last=True,
		pin_memory=True
	)
	# valid_loader = DataLoader(
	# 	valid_data,
	# 	num_workers=args.num_workers,
	# 	batch_size=args.batch_size,
	# 	shuffle=True,
	# 	drop_last=True,
	# 	pin_memory=True
	# )

	# train_iterator = iter(train_loader)
	# valid_iterator = iter(valid_loader)

	return train_loader


