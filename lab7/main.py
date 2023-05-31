import numpy as np
from evaluator import evaluation_model
from dataloader_b import iclevrLoader
import json
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
import csv
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.optim.lr_scheduler import StepLR
from diffusers import UNet2DConditionModel, UNet2DModel
from torch.utils.data import SubsetRandomSampler
import random
import math
import os
from accelerate import Accelerator
#from dataloader_b import encoding_dict
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers import DiffusionPipeline
import torchvision
import time
# import sys
# sys.setrecursionlimit(30000)  # Set the recursion limit to a higher value
# caculate alpha
timestep = 1200
beta = torch.linspace(1e-4, .002, timestep) #recommend 1e-4 to .002, timespace=2000
alpha = 1 - beta
alpha_cumprod = torch.cumprod(alpha, dim=0)
sqrt_alpha_cumprod = torch.sqrt(alpha_cumprod)
sqrt_oneminus_alpha_cumprod = torch.sqrt(1 - alpha_cumprod)

one_minus_alpha_cumprod_t_minus_1 =  torch.cat((torch.tensor(1).unsqueeze(0), (1 - alpha_cumprod)[:-1]))
one_minus_alpha_cumprod = (1 - alpha_cumprod)
sqrt_variance =  torch.sqrt((beta * (one_minus_alpha_cumprod_t_minus_1/one_minus_alpha_cumprod)))
#print(sqrt_variance.shape)


def compute_xt(args, data, rand_t, noise):
    # caculate coef
    coef_x0 = []
    coef_noise = []
    # select coef
    for i in range(data.shape[0]):
        coef_x0.append(sqrt_alpha_cumprod[rand_t[i]-1])
        coef_noise.append(sqrt_oneminus_alpha_cumprod[rand_t[i]-1])
    coef_x0 = torch.tensor(coef_x0)
    coef_noise = torch.tensor(coef_noise)
    # return xt
    #print(coef_x0.shape, data.shape, coef_noise.shape, noise.shape)
    coef_x0 = coef_x0[:, None, None, None]
    coef_noise = coef_noise[:, None, None, None]
    #print(coef_x0.shape, data.shape, coef_noise.shape, noise.shape)
    return coef_x0.to(args.device) * data.to(args.device) + coef_noise.to(args.device) * noise.to(args.device)


def train(args, model, device, train_loader, optimizer, epoch, lr_scheduler, accelerator):
    model.train()          

    for batch_idx, (data, cond) in enumerate(train_loader):
        #print(data.shape[0])
        data, cond = data.to(device,dtype=torch.float32), cond.to(device)
        cond = cond.squeeze()
        optimizer.zero_grad()
        # select t
        rand_t = torch.tensor([random.randint(1, timestep) for i in range(data.shape[0])])
        # select noise
        noise = torch.randn(data.shape[0], 3, 64, 64)
        xt = compute_xt(args, data, rand_t, noise)
        ''' 
        Model usage
        # sample: FloatTensor
        # timestep: typing.Union[torch.Tensor, float, int]
        # class_labels: typing.Optional[torch.Tensor] = None
        # return_dict: bool = True 
        # output shape = (batch_size, num_channels, height, width))
        '''
        ''''''
        output = model(sample = xt.to(args.device), timestep = rand_t.to(args.device), class_labels  = cond.to(torch.float32).to(args.device)) 
        loss = nn.MSELoss()(output.sample.to(args.device), noise.to(args.device))
        accelerator.backward(loss)
        lr_scheduler.step()
        optimizer.step()
        ''''''
        #lr_scheduler.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.10f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                ((100 * batch_idx) / len(train_loader)), loss.item()))
    return

def compute_prev_x(xt, t, pred_noise, args):
    coef = 1/torch.sqrt(alpha[t-1])
    noise_coef = beta[t-1] / sqrt_oneminus_alpha_cumprod[t-1]
    if t <= 1 :
        z = 0
    else:
        z = torch.randn(args.test_batch, 3, 64, 64)
    sqrt_var = sqrt_variance[t-1] 
    mean = coef * (xt - noise_coef * pred_noise)
    #print(type(mean), type(sqrt_var), type(z))
    prev_x = mean.to("cpu") + sqrt_var.to("cpu") * z
    return prev_x


def save_images(images, name):
    #print(images[0])
    grid = torchvision.utils.make_grid(images)
    save_image(grid, fp = "./"+name+".png")

def sample(model, device, test_loader, args, filename):
    # denormalize
    transform=transforms.Compose([
            transforms.Normalize((0, 0, 0), (1/0.5, 1/0.5, 1/0.5)),
            transforms.Normalize((-0.5, -0.5, -0.5), (1, 1, 1)),
        ])
    model.eval()
    xt = torch.randn(args.test_batch, 3, 64, 64)
    with torch.no_grad():
        for batch_idx, (img, cond) in enumerate(test_loader):
            cond = cond.to(device)
            # transform one-hot to embed's input
            # cond = transform_code(cond_onehot)
            cond = cond.squeeze()
            # print(cond)
            # print(cond, cond.shape)
            # print(cond_onehot, cond_onehot.shape)
            for t in range(timestep, 0, -1):
                # pred noise
                output = model(sample = xt.to(args.device), timestep = t, class_labels = cond.to(torch.float32).to(args.device))
                # compute xt-1
                xt = compute_prev_x(xt.to(args.device), t, output.sample.to(args.device), args)

            # evaluate
            evaluate = evaluation_model()
            acc = evaluate.eval(xt.to(args.device), cond.to(args.device), filename)
            torch.save(xt, f = filename+".pt")
            print("Test Result:", acc)
            with open('{}.txt'.format(filename), 'a') as test_record:
                test_record.write(('Accuracy : {}\n'.format(acc)))
            # denormalize
            img = transform(xt)
            save_images(img, name=filename)
            

        

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('-d', '--device', default='cuda')
    parser.add_argument('--train_batch', type=int, default=20)
    parser.add_argument('--test_batch', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--lr', type=float, default=1e-4 * 0.5)
    parser.add_argument('--gamma', type=float, default=0.7)
    parser.add_argument('--log-interval', type=int, default=10)
    parser.add_argument('--save-model', action='store_true', default=True)



    args = parser.parse_args()
    device = torch.device(args.device)
    train_kwargs = {'batch_size': args.train_batch}
    test_kwargs = {'batch_size': args.test_batch}

    #manipulate data
    train_loader = torch.utils.data.DataLoader(iclevrLoader(root="./dataset/", mode="train"),**train_kwargs,shuffle=True)
    print(len(train_loader.dataset))
    test_loader = torch.utils.data.DataLoader(iclevrLoader(root="./dataset/", mode="test"),**test_kwargs,shuffle=False)
    test_loader_new = torch.utils.data.DataLoader(iclevrLoader(root="./dataset/", mode="new_test"),**test_kwargs,shuffle=False)
    #create model
    model = UNet2DModel(
        sample_size = 64,
        in_channels = 3,
        out_channels = 3,
        layers_per_block = 2,
        class_embed_type = None,
        #num_class_embeds = 2325, #C (24, 3) + 1
        block_out_channels = (128, 128, 256, 256, 512, 512), 
        down_block_types=(
        "DownBlock2D",  # a regular ResNet downsampling block
        "DownBlock2D",
        "DownBlock2D",
        "DownBlock2D",
        "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
        "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",  # a regular ResNet upsampling block
            "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
    )
    model = model.to(args.device)
    model.class_embedding = nn.Linear(24 ,512)
    #optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=len(train_loader) * 500,
    )

    #load model
    model = UNet2DModel.from_pretrained(pretrained_model_name_or_path ="local-unetepoch_15",  
        variant="non_ema", from_tf=True,low_cpu_mem_usage=False,ignore_mismatched_sizes=True)
    print(model.class_embedding)
    model.class_embedding = nn.Linear(24 ,512)
    state_dict = torch.load("local-unetepoch_15/diffusion_pytorch_model.non_ema.bin")
    filtered_state_dict = {k[16:]: v for k, v in state_dict.items() if k =="class_embedding.weight" or k=="class_embedding.bias"}
    model.class_embedding.load_state_dict(filtered_state_dict)
    print(model.class_embedding)
    model = model.to(args.device)

    #print("==test.json==")
    # sample(model, device, test_loader, args, "unettest")
    # os._exit()
    #sample(model, device, test_loader, args, "test_")

    # Accelerator
    accelerator = Accelerator()
    model, optimizer, train_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_loader, lr_scheduler
    )

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch, lr_scheduler, accelerator)
        if args.save_model and (epoch%1)==0 :
            if args.save_model and (epoch%5)==0 :
                model.save_pretrained("./local-unet"+"epoch_"+str(epoch), variant="non_ema")
            print("==test.json==")
            sample(model, device, test_loader, args, "test_"+str(epoch))
            # print("==test_new.json==")
            # sample(model, device, test_loader_new, args, "new_test_"+str(epoch))
            # model.save_pretrained("./local-unet"+"epoch_"+str(epoch), variant="non_ema")

    

if __name__ == '__main__':
    main()