"""
PyTorch Distributed-DataParallel Example Script for Distributed Training Across Multiple GPUs
"""

# (0) prerequisite packages 
import torch
import torch.nn.functional as F
from torch.autograd import grad
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

"""
The following code should be in addition to the code and NN model used in the 'Introduction to PyTorch' notebook
"""

# (1) pytorch utilities for distributed training 
import os
# use .spawn multiple processes and apply a function to multiple inputs in parallel
import torch.multiprocessing as mp 
# divide the dataset for each GPU
from torch.utils.data.distributed import DistributedSampler
# main class for DDP
from torch.nn.parallel import DistributedDataParallel as DDP
# initialize and quit the distributed training mods
from torch.distributed import init_process_group, destroy_process_group 

"""
! To compute below, insert necessary code here from notebook here !
"""

# (2) model training with the DistributedDataParallel() strategy 

def ddp_setup(rank, world_size):
    # address of the main node 
    os.environ["MASTER_ADDR"] = "localhost"

    # any free port on the machine 
    os.environ["MASTER_PORT"] = "12345"
    init_process_group(
        backend="nccl", # NVIDIA Collective Communication Library 
        rank=rank,
        world_size=world_size # number of GPUs to use 
    )
    torch.cuda.set_device(rank)

# sets the current GPU device on which tensors will be allocated and operations will be performed 
def prepare_dataset():
    # insert dataset preparation code 
    train_loader = DataLoader(
        dataset=train_ds,
        batch_size=2,
        shuffle=False, # taken care of by DistributedSampler for now
        pin_memory=True,
        sampler=DistributedSampler(train_ds)
    )
    return train_loader, test_loader

# main function running the model training 
def main(rank, world_size, num_epochs):
    ddp_setup(rank, world_size)
    train_loader, test_loader = prepare_dataset()
    model = NeuralNetwork(num_inputs=2, num_outputs=2)
    model.to(rank)
    optimizer = torch.optim.SGD(model.parameters(), lr=.5)
    model = DDP(model, device_ids=[rank]) # rank = GPU ID
    for epoch in range(num_epochs):
        for features, labels in train_loader:
            features, labels = features.to(rank), labels.to(rank)
            # insert model prediction and backpropagation code
            print(f"[GPU{rank}] Epoch: {epoch+1:03d}/{num_epochs:03d}"
                  f" | Batchsize {labels.shape[0]:03d}"
                  f" | Train/Val Loss: {loss: .2f}"
                  )
    model.eval()
    train_acc = compute_accuracy(model, train_loader, device=rank)
    if rank==0:
        print(f"[GPU{rank}] Training accuracy", train_acc)
    test_acc = compute_accuracy(model, test_loader, device=rank)
    if rank==0:
        print(f"[GPU{rank}] Test accuracy", test_acc)
    destroy_process_group() # cleans up resource allocation

# prelaunch equivalence/action clause 
if __name__ == "__main__":
    print("Number of GPUs available:", torch.cuda.device_count())
    torch.manual_seed(123) # reproducibility 
    num_epochs = 3
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, num_epochs), nprocs=world_size) 
    # launches the main function using multiple processes, where nprocs=world_size means one process per GPU
