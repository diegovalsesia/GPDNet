#!/bin/bash

models=("GPDNet_mse_sp")
sigmas=("0.02")
nn="16nn"

this_folder=`pwd`

for model in ${models[@]}; do
        
	for sigma in ${sigmas[@]}; do

		save_dir="$this_folder/Results/$model/$sigma/$nn/saved_models/"
		denoised_dir="$this_folder/Results/$model/$sigma/$nn/point_clouds"
		mkdir -p $denoised_dir

		CUDA_VISIBLE_DEVICES=2 python "Code/$model/test_conv_general.py" --save_dir $save_dir --denoised_dir $denoised_dir

	done
done

