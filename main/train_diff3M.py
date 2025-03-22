import argparse
import sys
import torch
import os
sys.path.append("..")
sys.path.append(".")
from utils.sub_util import (
    add_dict_to_argparser,
    args_to_dict
)
from utils.model_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
)
from utils.resample import create_named_schedule_sampler
from utils import dist_util, logger
from dataloader.mimicloader import MimicDataset
from utils.train_util import TrainLoop #train_util
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist


def create_argparser():
    defaults = dict(
        data_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=100,
        save_interval=70000,
        resume_checkpoint='',
        use_fp16=False,
        fp16_scale_growth=1e-3,
        dataset='chexpert',
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    # print(parser.parse_args())
    add_dict_to_argparser(parser, defaults)
    return parser

def main():
    args = create_argparser().parse_args()

    if not os.path.exists('./results/' + args.exp_name):
        os.mkdir('./results/' + args.exp_name)

    print(args)
    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")

    model, diffusion = create_model_and_diffusion( # model: reverse process (learnable), diffusion: diffusion architecture
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion,  maxt=1000)
    
    logger.log("creating data loader ...")

    if args.dataset == 'mimic':
        assert args.one_class == True
        ds = MimicDataset(args.data_dir, args.image_size, is_test=False)
        sampler = DistributedSampler(ds, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=True)
        datal = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False, sampler=sampler)
    else:
        exit('dataset name error')

    logger.log('Training ...')

    TrainLoop(
        model = model,
        diffusion = diffusion,
        data = datal,
        batch_size = args.batch_size,
        microbatch = args.microbatch,
        lr = args.lr,
        ema_rate = args.ema_rate,
        log_interval = args.log_interval,
        save_interval = args.save_interval,
        resume_checkpoint = args.resume_checkpoint,
        use_fp16 = args.use_fp16,
        fp16_scale_growth = args.fp16_scale_growth,
        schedule_sampler = schedule_sampler,
        weight_decay = args.weight_decay,
        lr_anneal_steps = args.lr_anneal_steps,
        dataset = args.dataset,
        exp_name = args.exp_name
    ).run_loop()






if __name__ == "__main__":
    main()