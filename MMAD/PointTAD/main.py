# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

import argparse
import os
import datetime
import json
import random
import time
from pathlib import Path
import pandas as pd
from ruamel import yaml

import matplotlib.pyplot as plt
import numpy as np
import torch
from ruamel.yaml import YAML
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist

from datasets.dataset import collate_fn
import util.misc as utils
from datasets import build_dataset
from datasets.evaluate import evaluate_mAP, multi_formatting
from engine import evaluate, train_one_epoch
from models import build_model

import warnings

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


warnings.filterwarnings("ignore", category=UserWarning)

def main(args):
    utils.init_distributed_mode(args)
    # print('git:\n  {}\n'.format(utils.get_sha()))
    # print(args)
    # print(f"**************{args.dataset}*************")
    if args.dataset not in ['mmad']:
        print(f"DataSet {args.dataset} Not Implemented\n")
        raise NotImplementedError

    device = torch.device(args.device)
    print(device)
    print(torch.__version__)

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion, postprocessors = build_model(args)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters()
                       if p.requires_grad)
    print('number of params:', n_parameters)

    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    
    optimizer = torch.optim.AdamW(param_dicts,
                                  lr=args.lr,
                                  weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop, gamma=0.5)

    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)
    dataset_test = build_dataset(image_set='test', args=args)

    print("len dataset test:", len(dataset_test))

    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
        sampler_test = DistributedSampler(dataset_test, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        sampler_test = torch.utils.data.SequentialSampler(dataset_test)

    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train,
                                                        args.batch_size,
                                                        drop_last=True)


    data_loader_train = DataLoader(dataset_train,
                                   batch_sampler=batch_sampler_train,
                                   collate_fn=collate_fn,
                                   prefetch_factor=2,
                                   persistent_workers=True,
                                   num_workers=1)

    data_loader_val = DataLoader(dataset_val,
                                 args.batch_size,
                                 sampler=sampler_val,
                                 drop_last=False,
                                 prefetch_factor=2,
                                 persistent_workers=True,
                                 collate_fn=collate_fn,
                                 num_workers=10)

    data_loader_test = DataLoader(dataset_test,
                                 args.batch_size,
                                 sampler=sampler_test,
                                 drop_last=False,
                                 prefetch_factor=2,
                                 persistent_workers=True,
                                 collate_fn=collate_fn,
                                 num_workers=10)

    print("dataset train", len(dataset_train))

    output_dir = Path(args.output_dir)
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(args.resume,
                                                            map_location='cpu',
                                                            check_hash=True)
        else:
            print(("=> loading checkpoint '{}'".format(args.resume)))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            pretrained_dict = checkpoint['model']
            # only resume part of model parameter
            model_dict = model_without_ddp.state_dict()
            pretrained_dict = {
                k: v
                for k, v in pretrained_dict.items() if k in model_dict
            }
            model_dict.update(pretrained_dict)
            model_without_ddp.load_state_dict(model_dict)
            print(("=> loaded '{}' (epoch {})".format(args.resume,
                                                      checkpoint['epoch'])))

    if args.load:
        checkpoint = torch.load(args.load, map_location='cpu')
        args.start_epoch = checkpoint['epoch'] + 1
        model_without_ddp.load_state_dict(checkpoint['model'])

    if args.eval:
        if args.eval_set == "val":
            data_loader_evaluation = data_loader_val
        else:
            data_loader_evaluation = data_loader_test
        evaluator, eval_loss_dict = evaluate(model, criterion, postprocessors,
                                             data_loader_evaluation, device, args)
        res = evaluator.summarize()
        if utils.get_rank() == 0:
            results_pd = multi_formatting(res, args.dataset)
            results_pd.to_csv(args.output_dir + 'results_{}.csv'.format(args.eval_set))
            results_read = pd.read_csv(args.output_dir + 'results_{}.csv'.format(args.eval_set))
            test_stats = evaluate_mAP(results_read, args.dataset, args.num_classes, sub_set=args.eval_set)
            print(test_stats)
        return

    print('Start training')
    start_time = time.time()
    best_map = 0
    best_epoch = 0

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)

        train_stats, train_loss_dict = train_one_epoch(model, criterion,
                                                       data_loader_train,
                                                       optimizer, device,
                                                       epoch, args)
        lr_scheduler.step()
        if args.output_dir:
            checkpoint_paths = [output_dir / f'{args.dataset.capitalize()}_checkpoint.pth']
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master(
                    {
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        'args': args,
                    }, checkpoint_path)

        evaluator, eval_loss_dict = evaluate(model, criterion, postprocessors,
                                             data_loader_val, device, args)
        res = evaluator.summarize()
        
        if utils.get_rank() == 0:
            results_pd = multi_formatting(res, args.dataset)
            results_pd.to_csv(f'outputs/{args.dataset.capitalize()}_results_epoch_{epoch}.csv')
            results_pd.to_csv(f'outputs/{args.dataset.capitalize()}_results.csv')
            results_read = pd.read_csv(f'outputs/{args.dataset.capitalize()}_results.csv')
            test_stats = evaluate_mAP(results_read, args.dataset, args.num_classes, sub_set='val')

            log_stats = {
                **{f'train_{k}': v
                   for k, v in train_stats.items()},
                **{f'mAP': test_stats}, 'epoch': epoch,
                'n_parameters': n_parameters}

            if (float(test_stats) > best_map):
                best_map = float(test_stats)
                best_epoch = epoch
                with (output_dir / f'{args.dataset.capitalize()}_log_best_map_epoch_{epoch}.txt').open('w') as f:
                    f.write(json.dumps(log_stats) + '\n')
                checkpoint_path = output_dir / f'{args.dataset.capitalize()}_checkpoint_best_map.pth'
                utils.save_on_master(
                    {
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        'args': args,
                    }, checkpoint_path)


            if args.output_dir and utils.is_main_process():
                with (output_dir / f'{args.dataset.capitalize()}_log.txt').open('a') as f:
                    f.write(json.dumps(log_stats) + '\n')

        dist.barrier()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    print('best map: ', best_map)
    print('best epoch:', best_epoch)

def get_args_parser():
    parser = argparse.ArgumentParser('PointTAD detector',
                                     add_help=False)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--dataset', default='mmad')
    parser.add_argument('--output_dir', default='outputs/', help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--load', default='', help='load checkpoint')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=1, type=int, metavar='N', help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--eval_set', default='test', help='evaluation set')
    parser.add_argument('--img_tensor', action='store_true')
    parser.add_argument('--dense_result', action='store_true')
    parser.add_argument('--num_workers', default=8, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser

def get_configs(dataset):
    yaml = YAML(typ='rt')
    default_config = yaml.load(open('datasets/dataset_cfg.yaml', 'r'))[dataset]
    return default_config

if __name__ == '__main__':
    os.environ[
        "TORCH_DISTRIBUTED_DEBUG"
    ] = "DETAIL"
    parser = argparse.ArgumentParser('PointTAD training and evaluation script',
                                     parents=[get_args_parser()])
    args = parser.parse_args()
    configs = get_configs(args.dataset)
    args.__dict__.update(configs)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
