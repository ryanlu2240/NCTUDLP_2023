import os
import copy
import numpy as np
import json
import torch
import argparse
import torch.nn as nn
from tqdm import tqdm
from torch import optim
from utils import *
from modules import UNet_conditional, EMA
import logging
from torch.utils.tensorboard import SummaryWriter
from loader import load_train_data


class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=(240, 320), device="cuda:1"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        self.img_size = img_size
        self.device = device

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, n, labels, cfg_scale=3):
        logging.info(f"Sampling {n} new images....")
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 3, self.img_size[0], self.img_size[1])).to(self.device)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t, labels)
                if cfg_scale > 0:
                    uncond_predicted_noise = model(x, t, None)
                    predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x

def train(args ,train_dataloader):
    device = args.device
    dataloader = train_dataloader
    model = UNet_conditional(num_classes=args.num_classes, device=device).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()
    diffusion = Diffusion(img_size=(64, 64), device=device)
    l = len(dataloader)
    ema = EMA(0.995)
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)

    for epoch in tqdm(range(args.epochs + 1)):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader)
        for i, (images, labels) in enumerate(pbar):
            images = images.to(device)
            labels = labels.to(device)
            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(images, t)
            if np.random.random() < 0.1:
                labels = None
            predicted_noise = model(x_t, t, labels)
            loss = mse(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema.step_ema(ema_model, model)

            pbar.set_postfix(MSE=loss.item())

        if epoch % 50 == 0:
            # testing
            #test
            test_labels = []
            obj_data = json.load(open(args.objects_json))
            data = json.load(open(args.test_data_json))
            output_img = []
            output_img_ema = []
            for i in data:
                onehot = torch.zeros((1,24))
                for cls in i:
                    onehot[0][obj_data[cls]] = 1
                test_labels.append(onehot)

                onehot = onehot.float().to(device)
                sampled_images = diffusion.sample(model, n=len(onehot), labels=onehot)
                ema_sampled_images = diffusion.sample(ema_model, n=len(onehot), labels=onehot)
                # print(sampled_images.shape)
                output_img.append(sampled_images[0])
                output_img_ema.append(ema_sampled_images[0])
            save_images(output_img, os.path.join("result64", "test", f"{epoch}.jpg"))
            save_images(output_img_ema, os.path.join("result64",  "test", f"{epoch}_ema.jpg"))

            #new_test
            newtest_labels = []
            newoutput_img = []
            newoutput_img_ema = []
            obj_data = json.load(open(args.objects_json))
            data = json.load(open(args.newtest_data_json))
            for i in data:
                onehot = torch.zeros((1, 24))
                for cls in i:
                    onehot[0][obj_data[cls]] = 1
                newtest_labels.append(onehot)

                onehot = onehot.float().to(device)
                sampled_images = diffusion.sample(model, n=len(onehot), labels=onehot)
                ema_sampled_images = diffusion.sample(ema_model, n=len(onehot), labels=onehot)
                # print(sampled_images.shape)
                newoutput_img.append(sampled_images[0])
                newoutput_img_ema.append(ema_sampled_images[0])
            save_images(newoutput_img, os.path.join("result64", "new_test", f"{epoch}.jpg"))
            save_images(newoutput_img_ema, os.path.join("result64",  "new_test", f"{epoch}_ema.jpg"))


            torch.save(model.state_dict(), os.path.join("models/epoch", f"{epoch}_ckpt64.pt"))
            torch.save(ema_model.state_dict(), os.path.join("models/epoch", f"{epoch}_ema_ckpt.pt"))
            torch.save(optimizer.state_dict(), os.path.join("models/epoch", f"{epoch}_optim64.pt"))


def parse_args():
    parser = argparse.ArgumentParser()

	## What to do
    parser.add_argument("--train", default=False, action="store_true")
    parser.add_argument("--test" , default=False, action="store_true")

	## Hyper-parameters

    parser.add_argument("--seed"      , default=1     , type=int,    help="manual seed")
    parser.add_argument("--num_workers"      , default=4     , type=int)
    parser.add_argument("--batch_size"      , default=6    , type=int)
    parser.add_argument("--lr"        , default=3e-4 , type=float,  help="learning rate")
    parser.add_argument("--device"    , type=str      , default="cuda:1")
    parser.add_argument("--dataset_root", type=str, default='./iclevr/')
    parser.add_argument("--train_data_json", type=str, default='./TA_dataset/train.json')
    parser.add_argument("--test_data_json", type=str, default='./TA_dataset/test.json')
    parser.add_argument("--newtest_data_json", type=str, default='./TA_dataset/new_test.json')
    parser.add_argument("--objects_json", type=str, default='./TA_dataset/objects.json')
    parser.add_argument("--num_classes"      , default=24     , type=int)

    parser.add_argument("--epochs"      , default=500     , type=int)
    
    
    args = parser.parse_args()
    return args



if __name__ == '__main__':
    args = parse_args()

    train_dataloader = load_train_data(args)
    train(args, train_dataloader)



