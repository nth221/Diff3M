"""
Helpers for distributed training.
"""

import io
import os
import socket

import blobfile as bf
#from mpi4py import MPI
import torch as th
import torch.distributed as dist
from datetime import timedelta

# Change this to reflect your cluster layout.
# The GPU for a given rank is (rank % GPUS_PER_NODE).
GPUS_PER_NODE = 8

SETUP_RETRY_COUNT = 3


def setup_dist():
    """
    Setup a distributed process group.
    """
    if dist.is_initialized():
        return

    rank = int(os.environ.get("RANK"))
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank % GPUS_PER_NODE)#f"{MPI.COMM_WORLD.Get_rank() % GPUS_PER_NODE}"

    #comm = MPI.COMM_WORLD
  #  print('commworld, 'f"{MPI.COMM_WORLD.Get_rank() % GPUS_PER_NODE}", comm)
    backend = "gloo" if not th.cuda.is_available() else "nccl"
   # print('commrank', comm.rank)
   # print('commsize', comm.size)

    if backend == "gloo":
        hostname = "localhost"
    else:
        hostname = socket.gethostbyname(socket.getfqdn())
    # os.environ["MASTER_ADDR"] = socket.gethostbyname(socket.gethostname())
    os.environ["MASTER_ADDR"] = "127.0.0.1"

    # print('*****', socket.gethostbyname(socket.gethostname()))
    os.environ["RANK"] = os.environ.get("RANK") #str(comm.rank)
    os.environ["WORLD_SIZE"] = os.environ.get("WORLD_SIZE") #str(comm.size)
    print(os.environ["RANK"])  
    print(os.environ["WORLD_SIZE"])
   
   # print('commmasteraddr', comm.bcast(hostname, root=0))

   # port = comm.bcast(_find_free_port(), root=0)
   # print('port', port)
   # os.environ["MASTER_PORT"] = str(port)

    # s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # s.bind(("", 0))
    # s.listen(1)
    # port = s.getsockname()[1]
    # s.close()
    # print('port2', port)
    # os.environ["MASTER_PORT"] = str(port)
    os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT")
    print(os.environ["MASTER_PORT"])
    dist.init_process_group(backend=backend, init_method="env://", timeout=timedelta(minutes=30))


def dev():
    """
    Get the device to use for torch.distributed.
    """
    if th.cuda.is_available():
        return th.device(f"cuda")
    return th.device("cpu")


def load_state_dict(path, **kwargs):
    """
    Load a PyTorch file without redundant fetches across MPI ranks.
    """
    #print('mpicommworldgetrank', MPI.COMM_WORLD.Get_rank())
    mpigetrank=0
   # if MPI.COMM_WORLD.Get_rank() == 0:
    if mpigetrank==0:
        with bf.BlobFile(path, "rb") as f:
            data = f.read()
    else:
        data = None
   # data = MPI.COMM_WORLD.bcast(data)
  #  print('mpibacst', MPI.COMM_WORLD.bcast(data))
    return th.load(io.BytesIO(data), **kwargs)


def sync_params(params):
    """
    Synchronize a sequence of Tensors across ranks from rank 0.
    """
    for p in params:
        with th.no_grad():
            dist.broadcast(p, 0)


def _find_free_port():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    finally:
        s.close()
