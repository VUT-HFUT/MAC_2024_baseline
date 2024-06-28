# mmad training script

CUDA_VISIBLE_DEVICES=0,1,2 python -m torch.distributed.launch --nproc_per_node=3 --master_port=11302 --use_env main.py --dataset mmad --img_tensor