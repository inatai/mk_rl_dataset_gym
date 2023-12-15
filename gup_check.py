import torch

def get_device(gpu_id=-1):
    if gpu_id >= 0 and torch.cuda.is_available():
        return torch.device("cuda", gpu_id)
    else:
        return torch.device("cpu")
    
    
device = get_device()
print(device)  # cpu

device = get_device(gpu_id=0)
print(device)  # cuda:0