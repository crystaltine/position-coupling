cd ../..

n_train=20
n_test=50
n_layers=6
n_heads=8
d_model=512
d_ff=2048
d_kv=$((d_model/n_heads))

python run_parallel.py \
    --group_name Parity_${n_train}_${n_test} \
    --exp_name NoPE_${n_layers}layers_${n_heads}heads \
    --seeds 0 1 2 3 \
    --seeds_data 0 1 \
    --devices 0 1 2 3 4 5 6 7 \
    --num_exp_per_device 8 \
    --overrides \
        model.position_encoding_type=none \
        model.num_layers=$n_layers \
        model.num_heads=$n_heads \
        model.normalization_layer=rmsnorm \
        model.layer_norm_position=pre_post \
        model.feed_forward_proj=gated-gelu \
        model.d_model=$d_model \
        model.d_ff=$d_ff \
        model.d_kv=$d_kv \
        model.save=True \
        task=parity \
        task.train.min_n_digits=1 \
        task.train.max_n_digits=$n_train \
        task.train.n_data=10000 \
        task.val.min_n_digits=$n_train \
        task.val.max_n_digits=$n_train \
        task.val.n_data=1000 \
        task.val_long.min_n_digits=$n_test \
        task.val_long.max_n_digits=$n_test \
        task.val_long.n_data=1000 \
        training.batch_size_train=20 \
        training.batch_size_eval=100 \
        training.n_steps=50000 \
        training.optimizer.lr=0.00003 \
        training.optimizer.weight_decay=0 \

