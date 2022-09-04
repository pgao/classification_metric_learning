#!/bin/bash

cd ~/classification_metric_learning/

# python3 metric_learning/train_classification.py \
# 	--dataset StanfordOnlineProducts \
# 	--dim 2048 \
# 	--model_name resnet50 \
# 	--epochs_per_step 15 \
# 	--num_steps 2 \
# 	--test_every_n_epochs 5 \
# 	--lr 0.01 \
# 	--lr_mult 1 \
# 	--class_balancing \
# 	--images_per_class 1 \
# 	--batch_size 16 \
# 	--output /home/pgao/classification_metric_learning/model_outputs_stanford_test_sgd

python3 metric_learning/train_classification.py \
	--dataset Aquarium \
	--dim 2048 \
	--model_name resnet50 \
	--epochs_per_step 15 \
	--num_steps 2 \
	--test_every_n_epochs 5 \
	--lr 0.01 \
	--lr_mult 1 \
	--class_balancing \
	--images_per_class 2 \
	--batch_size 16 \
	--output /home/pgao/classification_metric_learning/model_outputs_sgd_lr_tuned

cd ~/aquarium_hackathon_2022/
python3 run_end_to_end.py
cd ~/classification_metric_learning/
