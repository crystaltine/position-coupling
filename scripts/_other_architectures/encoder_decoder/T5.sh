cd ../../..

python run_parallel.py \
    --group_name Addition \
    --exp_name T5_revers \
    --seeds 0 \
    --devices 0 \
    --num_exp_per_device 1 \
    --overrides \
        model=T5 \