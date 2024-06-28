import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
 
from torch.nn.parallel import DistributedDataParallel as DDP
 
import argparse
 
class ToyModel(nn.Module):
    def __init__(self, h,layers):
        super(ToyModel, self).__init__()
        self.m = nn.Sequential(
            nn.Linear(10, h),
            *[nn.Linear(h, h) for _ in range(layers)],
            nn.Linear(h, 5)
        )
 
    def forward(self, x):
        return self.m(x)
 
 
def demo_basic(args):
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    print(f"Start running basic DDP example on rank {rank}.")
 
    # create model and move it to GPU with id rank
    device_id = rank % torch.cuda.device_count()
    model = ToyModel(args.h, args.layers).to(device_id)
    ddp_model = DDP(model, device_ids=[device_id])
    
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)
    
    for _ in range(args.epochs):
        optimizer.zero_grad()
        outputs = ddp_model(torch.randn(20, 10))
        labels = torch.randn(20, 5).to(device_id)
        loss_fn(outputs, labels).backward()
        optimizer.step()
        
        if rank==0:
            print(_)
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--h', type=int, default=1024, help='hidden points')
    parser.add_argument('--layers', type=int, default=10, help='MLP layer')
    parser.add_argument('--epochs', type=int, default=1000, help='epochs')
 
    args = parser.parse_args()
    
    print(vars(args))
    demo_basic(args)


# torchrun --nnodes=1 --nproc_per_node=1 test_dist.py