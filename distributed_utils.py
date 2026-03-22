import os
from datetime import timedelta
import torch
import torch.distributed as dist

def setup_ddp():
    if "LOCAL_RANK" not in os.environ:
        raise RuntimeError("LOCAL_RANK not found in environment. Please launch with torchrun.")

    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    # Add these two lines to bypass L40s PCIe P2P deadlocks
    os.environ["NCCL_P2P_DISABLE"] = "1"
    os.environ["NCCL_IB_DISABLE"] = "1"

    timeout_minutes = int(os.environ.get("DDP_TIMEOUT_MINUTES", "5"))
    dist.init_process_group(
        backend="nccl",
        timeout=timedelta(minutes=timeout_minutes),
        device_id=device 
    )
    
    return local_rank, global_rank, device

def cleanup_ddp():
    """Destroys the process group after training."""
    if dist.is_initialized():
        dist.destroy_process_group()

def is_main_process():
    """Returns True if the current process is the main one (Rank 0)."""
    return int(os.environ.get("RANK", 0)) == 0