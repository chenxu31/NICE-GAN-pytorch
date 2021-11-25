# code

## preprocessing
download code to Lian_project_code/
 
## train NICE-GAN
python main_pelvic.py --gpu {GPU_ID} --data_dir {DATA_DIR} --checkpoint_dir {CHECKPOINT_DIR}

## test NICE-GAN
python main_pelvic.py --phase test --gpu {GPU_ID} --data_dir {DATA_DIR} --checkpoint_dir {CHECKPOINT_DIR} --result_dir {OUTPUT_DIR}
