import argparse, os, sys, glob
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
from main import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from torch.utils.data import random_split, DataLoader, Dataset, Subset
from ldm.data.viton_daf import VitonDAFDataset
from ldm.data.viton import VitonDataset
from ldm.data.mpv import MPVDataset
import ldm.modules.image_degradation.utils_image as util



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        nargs="?",
        help="viton or mpv",
    )
    parser.add_argument(
        "--dataroot",
        type=str,
        nargs="?",
        help="dir containing image-mask pairs (`example.png` and `example_mask.png`)",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=200,
        help="number of ddim sampling steps",
    )

    parser.add_argument(
        "--configs",
        type=str,
        nargs="?",
        help="configs",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        nargs="?",
        help="checkpoint",
    )
    
    parser.add_argument(
        "--batchsize",
        type=int,
        default=1,
        nargs="?",
        help="batchsize",
    )

    opt = parser.parse_args()

    # create dataset
    if opt.dataset == "mpv":
        dataset = MPVDataset(opt.dataroot, datamode="test")
    elif opt.dataset == "viton_daf_pair":
        dataset = VitonDAFDataset(opt.dataroot, datamode="test", dataset_list='test_pairs.txt')
    elif opt.dataset == "viton_daf_unpair":
        dataset = VitonDAFDataset(opt.dataroot, datamode="test", dataset_list='test_unpairs.txt')
    elif opt.dataset == "viton_daf_muti_pos":
        dataset = VitonDAFDataset(opt.dataroot, datamode="test", dataset_list='test_multi_pos.txt', test_muti_pos = True)
    elif opt.dataset == "viton":
        dataset = VitonDataset(opt.dataorot, datamode="test")
    else:
        raise NotImplementedError

    dataloader = DataLoader(dataset, batch_size=opt.batchsize, shuffle=False)

    config = OmegaConf.load(opt.configs)
    model = instantiate_from_config(config.model)
    model.load_state_dict(torch.load(opt.checkpoint)["state_dict"], strict=False)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    model.eval()
    sampler = DDIMSampler(model)

    outdir = os.path.join(opt.outdir,opt.dataset)
    os.makedirs(outdir, exist_ok=True)
    psnr = []
    ssim = []
    with model.ema_scope():
        for data in dataloader:
            # print(data['c_name'])
            x, c = model.get_input(data, model.first_stage_key)
            shape = x.shape[1:]
            samples_ddim, _ = sampler.sample(S=opt.steps,
                                             conditioning=c,
                                             batch_size=c.shape[0],
                                             shape=shape,
                                             verbose=False)
            x_samples_ddim = model.decode_first_stage(samples_ddim)
            predicted_image = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
            # inpainted = (1-mask)*image+mask*predicted_image
            predicted_image = (predicted_image.cpu().numpy().transpose(0, 2, 3, 1) * 255).astype(np.uint8)

            image = ((data['image'].cpu().numpy() + 1.0) * 127.5).astype(np.uint8)
            masked_image = ((data['img_agnostic'].cpu().numpy() + 1.0) * 127.5).astype(np.uint8)
            target_cloth = ((data['cloth'].cpu().numpy() + 1.0) * 127.5).astype(np.uint8)
            
            N = min(len(data['c_name']),opt.batchsize)

            for i in range(N):
                outpath = os.path.join(outdir, data['img_name'][i][:-4]+data['c_name'][i])
                # Image.fromarray(np.concatenate((image, masked_image, target_cloth, predicted_image), axis=1)).save(outpath)
                Image.fromarray(predicted_image[i][:,32:224,:]).save(outpath)
                if opt.dataset == "viton_daf_pair":
                    psnr.append(util.calculate_psnr(predicted_image[i],image[i]))
                    ssim.append(util.calculate_ssim(predicted_image[i],image[i]))
            
    if opt.dataset == "viton_daf_pair":
        print(opt.dataset)
        print("psnr : ",np.mean(psnr))
        print("ssim : ",np.mean(ssim))
