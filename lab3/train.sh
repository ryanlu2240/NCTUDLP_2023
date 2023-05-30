for lr in 5e-3 2e-3 1e-3 5e-4 2e-4 1e-4
do
	python main.py --model EEGNet --lr "$lr" --save_plot
	python main.py --model DeepConvNet --lr "$lr" --save_plot
done