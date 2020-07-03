#!/bin/bash

models=("GPDNet_mse_sp")
sigmas=("0.02")
sig_val=("02")
nn="16nn"

this_folder=`pwd`
echo $this_folder
for model in ${models[@]}; do
        
	for sigma in ${sigmas[@]}; do
	
		# check if it is already trained or we want to retrain
		if [[ -f "$this_folder/log_dir/$model/$sigma/$nn/start_iter" ]]; then
			start_iter=`cat "$this_folder/log_dir/$model/$sigma/$nn/start_iter"`
		else
			start_iter=1
		fi
	       	echo $start_iter
		if [[ $start_iter -ge 20000000 ]]; then
			continue
		else
			train_data_file="$this_folder/Dataset/trainingset_data_patches.h5" 
			clean_val_data_file="$this_folder/Dataset/validationset_data_patches.h5"
      		noisy_val_data_file="$this_folder/Dataset/noisy_${sig_val}_validationset_data_patches.h5"
			log_dir="$this_folder/log_dir/$model/$sigma/$nn/"
			save_dir="$this_folder/Results/$model/$sigma/$nn/saved_models/"

			mkdir -p $log_dir
      
			CUDA_VISIBLE_DEVICES=2 python "Code/$model/main.py" --start_iter $start_iter --train_data_file $train_data_file --clean_val_data_file $clean_val_data_file --noisy_val_data_file $noisy_val_data_file --log_dir $log_dir --save_dir $save_dir
		fi

	done
done

