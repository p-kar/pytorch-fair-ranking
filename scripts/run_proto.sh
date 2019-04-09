source /scratch/cluster/pkar/pytorch-gpu-py3/bin/activate
code_root=__CODE_ROOT__

python -u $code_root/driver.py \
	--mode __MODE__ \
	--data_dir __DATA_DIR__ \
	--nworkers __NWORKERS__ \
	--bsize __BSIZE__ \
	--shuffle __SHUFFLE__ \
	--maxlen __MAXLEN__ \
	--dropout_p __DROPOUT_P__ \
	--hidden_size __HIDDEN_SIZE__ \
	--optim __OPTIM__ \
	--lr __LR__ \
	--wd __WD__ \
	--momentum __MOMENTUM__ \
	--epochs __EPOCHS__ \
	--max_norm __MAX_NORM__ \
	--start_epoch __START_EPOCH__ \
	--save_path __SAVE_PATH__ \
	--log_dir __LOG_DIR__ \
	--log_iter __LOG_ITER__ \
	--resume __RESUME__ \
	--seed __SEED__
