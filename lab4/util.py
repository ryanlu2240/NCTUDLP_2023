import pdb
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def save_exp(args, best):
	"""Save experiment results"""
	print("Saving training results...")
	with open("{}/best.txt".format(args.result_path), "a") as fw:
		fw.write(
			"{:8s}\t{:13.1E}\t{:10d}\t{:9.4f}\n".format(
				args.model, args.lr, best["epoch"], best["test_acc"]
			)
		)

def save_learning_curves(args, epochs, train_accs, test_accs):
	"""Write learning curves info"""
	print("Saving learning curves info...")
	with open("{}/learning_curves.txt".format(args.result_path), "a") as fw:
		pretrained_str = "w" if args.pretrained else "wo"
		
		fw.write(
			"{}\t{:2s} pretrained\t{}\t{}\t{}\n".format(
				args.model, pretrained_str, str(epochs), str(train_accs), str(test_accs)
			)
		)

def plot_confusion_matrix(args, labels, preds):
	"""
	Plot confusion matrix (Need to implement by myself!).
	Used when test only.
	"""
	print("Plotting confusion matrix...")

	## Calculate confusion matrix
	num_class = len(set(labels.tolist()))
	confusion_matrix = np.zeros((num_class, num_class))
	for idx in range(len(labels)):
		confusion_matrix[labels[idx]][preds[idx]] += 1

	## Normalize
	confusion_matrix = confusion_matrix / confusion_matrix.sum(axis=1)[np.newaxis].T

	## Plot
	textcolors = ("black", "white")
	fig, ax = plt.subplots()
	img = ax.imshow(confusion_matrix, cmap=plt.cm.Blues)
	for i in range(confusion_matrix.shape[0]):
		for j in range(confusion_matrix.shape[1]):
			ax.text(
				j, i, "{:.2f}".format(confusion_matrix[i, j]), 
				ha="center", va="center", 
				color=textcolors[confusion_matrix[i, j] > 0.5]
			)

	plt.colorbar(img)
	plt.title(f"Normalized Confusion Matrix (ResNet{args.model[-2:]})")
	plt.xlabel("Predicted Label")
	plt.ylabel("True Label")

	pretrained_str = "w" if args.pretrained else "wo"

	plt.savefig(f"{args.result_path}/cm_{args.model}_{pretrained_str}_pretrained.png")

def plot_learning_curves():
	"""Plot learning curves"""
	result_dict = {}
	with open("./result/learning_curves.txt") as f:
		for line in list(f.readlines())[-4:]:
			line = line.strip().rstrip()
			model, pretrained, epochs, train_accs, test_accs = line.split("\t")

			if model not in result_dict:
				result_dict[model] = {"w  pretrained": {}, "wo pretrained": {}}
			
			result_dict[model][pretrained]["epochs"] = np.array(eval(epochs)) + 1
			result_dict[model][pretrained]["train_accs"] = np.array(eval(train_accs))
			result_dict[model][pretrained]["test_accs" ] = np.array(eval(test_accs ))

	models = ["resnet18", "resnet50"]
	pretrained_strs = ["w  pretrained", "wo pretrained"]
	for model in models:
		plt.figure()
		for pretrained in pretrained_strs:
			epochs     = result_dict[model][pretrained]["epochs"]
			train_accs = result_dict[model][pretrained]["train_accs"]
			test_accs  = result_dict[model][pretrained]["test_accs" ]

			pretrained_str = "w/o" if "wo" in pretrained else "with"
			plt.plot(epochs[:20], train_accs[:20], label="Train ({} pretraining)".format(pretrained_str))
			plt.plot(epochs[:20],  test_accs[:20], label= "Test ({} pretraining)".format(pretrained_str))

		plt.legend()
		plt.xlabel("Epoch")
		plt.ylabel("Accuracy (%)")
		plt.xticks([5, 10, 15, 20])
		plt.title("Result Comparison (ResNet{})".format(model[-2:]))
		plt.savefig("./result/lr_{}.png".format(model))
		


if __name__ == "__main__":
	plot_learning_curves()



