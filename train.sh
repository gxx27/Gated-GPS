export CUDA_VISIBLE_DEVICES=0

python train.py --seed 22 \
    --batch_size 16 \
    --epoch 60 \
    --hidden_dim 128 \
    --esm_out 8 \
    --num_heads 8 \
    --num_layers 4 \
    --pe_dim 32 \
    --pe_ratio 0.2 \
    --lr 4e-4 \
    --weight_decay 4e-4 \
    --dropout 0.2 \
    --attn_dropout 0.7 \
    --act ReLU \
    --weight 0.3 \
    --alpha 0.36 \
    --beta 0.97 \
    --save_path ckpt \
    --use_esm 