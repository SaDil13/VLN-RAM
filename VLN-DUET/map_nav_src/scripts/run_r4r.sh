
train_alg=dagger

features=clip_l14
ft_dim=768
obj_features=vitbase
obj_ft_dim=768

ngpus=1
seed=0

name=${train_alg}-${features}
name=${name}-seed.${seed}
name=${name}-RAM

outdir=${DATA_ROOT}/R4R/exprs_map/finetune/${name}

flag="--root_dir ${DATA_ROOT}
      --dataset r4r
      --output_dir ${outdir}
      --world_size ${ngpus}
      --seed ${seed}
      --tokenizer bert      

      --enc_full_graph
      --graph_sprels
      --fusion dynamic

      --expert_policy spl
      --train_alg ${train_alg}
      
      --num_l_layers 9
      --num_x_layers 4
      --num_pano_layers 2
      
      --max_action_len 15
      --max_instr_len 200

      --batch_size 8
      --lr 1e-5
      --iters 100000
      --log_every 1000
      --optim adamW

      --features ${features}
      --image_feat_size ${ft_dim}
      --angle_feat_size 4

      --ml_weight 0.2   

      --feat_dropout 0.4
      --dropout 0.5
      
      --gamma 0."

# train stage 1
CUDA_VISIBLE_DEVICES='0' python r2r/main_nav.py $flag  \
      --tokenizer bert \
      --bert_ckpt_file 'put the R2R pretrained model (see pretrain_src) here' \
      --eval_first \
      --aug 'put the new instructions file here' \
      --train_aug_ft_file 'put the new feature file here' \


# train stage 2
CUDA_VISIBLE_DEVICES='0' python r2r/main_nav.py $flag  \
      --tokenizer bert \
      --resume_file 'put the stage1 best model here' \
      --eval_first 

# test
CUDA_VISIBLE_DEVICES='0' python r2r/main_nav.py $flag  \
      --tokenizer bert \
      --resume_file 'put your best model here' \
      --test --submit