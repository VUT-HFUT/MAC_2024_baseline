# mmad dataset. validation set.
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port=11102 --use_env main.py --dataset mmad --eval --eval_set val --load ./outputs/Mmad_checkpoint_best_map.pth
