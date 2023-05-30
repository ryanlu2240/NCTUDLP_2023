# for experiment in "model=resnet18 bs=32" "model=resnet50 bs=16"
# do
# 	eval $experiment

# 	# python main.py \
# 	# 	--train \
# 	# 	--seed 123 \
# 	# 	--model "$model" \
# 	# 	--pretrained \
# 	# 	--bs "$bs" \
# 	# 	--save_exp

# 	python main.py \
# 		--train \
# 		--seed 123 \
# 		--model "$model" \
# 		--bs "$bs" \
# 		--save_exp \
# 		--device cuda:1
# done


# Test ##
for model in resnet18 resnet50
do
	python main.py \
		--test \
		--seed 123 \
		--model "$model" \
		--bs 32 \
		# --save_confusion_matrix

	python main.py \
		--test \
		--seed 123 \
		--model "$model" \
		--pretrained \
		--bs 32 \
		# --save_confusion_matrix
done
# for experiment in "model=resnet18 bs=32" "model=resnet50 bs=16"
# do
# 	eval $experiment

# 	python main.py \
# 		--train \
# 		--seed 123 \
# 		--model "$model" \
# 		--bs "$bs" \
# 		--save_exp \
# 		--loss_weight
	
# 	#python main.py \
# 	#	--train \
# 	#	--seed 123 \
# 	#	--model "$model" \
# 	#	--pretrained \
# 	#	--bs "$bs" \
# 	#	--save_exp \
# 	#	--loss_weight
# done