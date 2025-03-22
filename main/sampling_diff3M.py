import matplotlib.pyplot as plt
import argparse
import os
from torchvision.transforms import ToPILImage
import csv

import sys
sys.path.append("..")
sys.path.append(".")
from dataloader.mimicloader import MimicDataset
import torch.nn.functional as F
import numpy as np
import torch 
import torch.distributed as dist
from utils import dist_util, logger
from utils.model_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
)
from utils.sub_util import (
    add_dict_to_argparser,
    args_to_dict
)
import matplotlib.pyplot as plt
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import pandas as pd
import timm

def visualize(img):
    _min = img.min()
    _max = img.max()
    normalized_img = (img - _min)/ (_max - _min)
    return normalized_img

def create_argparser():
    defaults = dict(
        data_dir="",
        clip_denoised=True,
        num_samples=10,
        batch_size=1,
        use_ddim=False,
        model_path="",
        classifier_path="",
        classifier_scale=100,
        noise_level=500,
        dataset='visa'
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


def main():
    args = create_argparser().parse_args()

    if not os.path.exists('./results/' + args.exp_name):
        os.mkdir('./results/' + args.exp_name)


    if not os.path.exists('./results/' + args.exp_name + '/heatmap'):
        os.mkdir('./results/' + args.exp_name + '/heatmap')

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    if args.dataset == 'mimic':
        assert args.one_class == True
        ds = MimicDataset(args.data_dir, args.image_size, is_test=True)
        sampler = DistributedSampler(ds, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=False)
        datal = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False, sampler=sampler)
        num_channels = 1
    else:
        exit('dataset name error ...')

   
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    if os.path.exists('./results/' + args.exp_name  + '/label_prediction.csv'):
        df = pd.read_csv('./results/' + args.exp_name  + '/label_prediction.csv', header=None)
        jpg_names = df.iloc[:, 0].tolist()
    else:
        jpg_names = []

    def model_fn(x, timesteps, demo, demo_cat, demo_indices, x_start, ehr, ehr_cat, ehr_indices):
        return model(x, timesteps, demo, demo_cat, demo_indices, x_start, ehr, ehr_cat, ehr_indices)

    logger.log("sampling...")
    all_images = []
    all_labels = []

    label_tensor = torch.tensor([])
    for idx, img in enumerate(datal):
        if img[-1][0].split('/')[-1] in jpg_names:
            print('skip')
            continue
        else:
            print(img[-1][0].split('/')[-1])

        model_kwargs = {}

        model_kwargs['demo'] = img[1].to(dist_util.dev())
        model_kwargs['demo_cat'] = img[2].to(dist_util.dev())
        model_kwargs['demo_indices'] = img[3].to(dist_util.dev())
        model_kwargs['x_start'] = img[0].to(dist_util.dev())

        model_kwargs['ehr'] = img[4].to(dist_util.dev())
        model_kwargs['ehr_cat'] = img[5].to(dist_util.dev())
        model_kwargs['ehr_indices'] = img[6].to(dist_util.dev())

        sample_fn = (
            diffusion.p_sample_loop_known if not args.use_ddim else diffusion.ddim_sample_loop_known
        )

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

        cond_fn = None
        sample, x_noisy, org = sample_fn(
            model_fn,
            (args.batch_size, num_channels, args.image_size, args.image_size), img, org=img,
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            cond_fn=cond_fn,
            device=dist_util.dev(),
            noise_level=args.noise_level,
            one_class = args.one_class
        )
        end.record()
        torch.cuda.synchronize()
        torch.cuda.current_stream().synchronize()


        print('time for 1000', start.elapsed_time(end))
        print('image name', img[8])

        generated = sample.squeeze(0)
        generated_to_pil = ToPILImage()

        generated_image = generated_to_pil(generated)
        generated_image.save('./results/' + args.exp_name  + '/heatmap/gen-' + str(int(img[7])) + '_' + img[8][0].split('/')[10])

        norm_sample = visualize(sample[0, 0, ...])
        norm_org = visualize(org[0, 0, ...])

        diff = abs(norm_org - norm_sample)

        not_norm_diff = abs(org[0, 0, ...] - sample[0, 0, ...])
        not_norm_max_diff = not_norm_diff.max()
        norm_max_diff = diff.max()

        mse = F.mse_loss(org[0, ...], sample[0, ...])
        norm_mse = F.mse_loss(norm_org, norm_sample)
        with open('./results/' + args.exp_name  + '/label_prediction.csv', mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow([img[8][0].split('/')[10], img[7].item(), mse.item(), norm_mse.item(), not_norm_max_diff.item(), norm_max_diff.item()])


    dist.barrier()
    logger.log("sampling complete")



if __name__ == "__main__":
    main()