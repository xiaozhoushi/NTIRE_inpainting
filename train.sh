python3 main.py \
        --data_path /mnt/bd/aurora-mtrc-sxz/datas/FFHQ/train \
        --validation_path /mnt/bd/aurora-mtrc-sxz/datas/FFHQ/val \
        --batch_size 16 \
        --train_epoch 100 \
        --image_size 256 \
	--DDP \
        --nodes 1 \
        --gpus 4 \
	--AMP
