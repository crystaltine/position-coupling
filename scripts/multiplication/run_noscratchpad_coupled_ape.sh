cd ../..

n_train=10
n_test=20
m_train=10
m_test=20

n_layers=6
n_heads=8

lr=0.00005
wd=0
d_model=1024
d_ff=2048
d_kv=$((d_model/n_heads)) 

# n_data=10000
n_data=500000
bs=250

maxpos_d=70
maxpos_o=35


## Multiplication Coupling, no scratchpad ##

python run_parallel.py \
    --group_name Multiplication_N${n_train}_${n_test}_M${m_train}_${m_test} \
    --exp_name coupled_noCoT_${n_layers}L${n_heads}H${d_model}dim_Data${n_data}BS${bs}LR${lr}WD${wd} \
    --seeds 0 1 2 3 \
    --seeds_data 0 1 \
    --devices 0 1 2 3 4 5 6 7 \
    --num_exp_per_device 1 \
    --overrides \
        model.position_encoding_type=abs_learned \
        model.num_layers=$n_layers \
        model.num_heads=$n_heads \
        model.normalization_layer=rmsnorm \
        model.layer_norm_position=pre_post \
        model.feed_forward_proj=gated-gelu \
        model.d_model=$d_model \
        model.d_ff=$d_ff \
        model.d_kv=$d_kv \
        model.n_positions=$((maxpos_d+1)) \
        model.save=True \
        task=multiplication_coupled \
        task.reverse_input=False \
        task.reverse_output=True \
        task.padding=True \
        task.max_position=$maxpos_d \
        task.train.min_n_digits_1=1 \
        task.train.max_n_digits_1=$n_train \
        task.train.min_n_digits_2=1 \
        task.train.max_n_digits_2=$m_train \
        task.train.n_data=$n_data \
        task.val.min_n_digits_1=1 \
        task.val.max_n_digits_1=$n_train \
        task.val.min_n_digits_2=1 \
        task.val.max_n_digits_2=$m_train \
        task.val.n_data=1000 \
        task.val_long_first.min_n_digits_1=$((n_train+1)) \
        task.val_long_first.max_n_digits_1=$n_test \
        task.val_long_first.min_n_digits_2=1 \
        task.val_long_first.max_n_digits_2=$m_train \
        task.val_long_first.n_data=1000 \
        task.val_long_second.min_n_digits_1=1 \
        task.val_long_second.max_n_digits_1=$n_train \
        task.val_long_second.min_n_digits_2=$((m_train+1)) \
        task.val_long_second.max_n_digits_2=$m_test \
        task.val_long_second.n_data=1000 \
        task.val_long.min_n_digits_1=$((n_train+1)) \
        task.val_long.max_n_digits_1=$n_test \
        task.val_long.min_n_digits_2=$((m_train+1)) \
        task.val_long.max_n_digits_2=$m_test \
        task.val_long.n_data=1000 \
        training.batch_size_train=$bs \
        training.batch_size_eval=20 \
        training.n_steps=50000 \
        training.optimizer.lr=$lr \
        training.optimizer.weight_decay=$wd




