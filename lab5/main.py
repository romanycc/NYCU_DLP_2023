import argparse
import itertools
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import bair_robot_pushing_dataset
from models.lstm import gaussian_lstm, lstm
from models.vgg_64 import vgg_decoder, vgg_encoder
from utils import init_weights, kl_criterion, plot_pred, finn_eval_seq, pred
import matplotlib.pyplot as plt
import pickle

torch.backends.cudnn.benchmark = True

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=0.002, type=float, help='learning rate')
    parser.add_argument('--beta1', default=0.9, type=float, help='momentum term for adam')
    parser.add_argument('--batch_size', default=12, type=int, help='batch size')
    parser.add_argument('--log_dir', default='./logs/fp', help='base directory to save logs')
    parser.add_argument('--model_dir', default='', help='base directory to save logs')
    parser.add_argument('--data_root', default='./sample_code/data', help='root directory for data')
    parser.add_argument('--optimizer', default='adam', help='optimizer to train with')
    parser.add_argument('--niter', type=int, default=300, help='number of epochs to train for')
    parser.add_argument('--epoch_size', type=int, default=600, help='epoch size')
    parser.add_argument('--tfr', type=float, default=1.0, help='teacher forcing ratio (0 ~ 1)')
    parser.add_argument('--tfr_start_decay_epoch', type=int, default=100, help='The epoch that teacher forcing ratio become decreasing')
    parser.add_argument('--tfr_decay_step', type=float, default=0, help='The decay step size of teacher forcing ratio (0 ~ 1)')
    parser.add_argument('--tfr_lower_bound', type=float, default=0, help='The lower bound of teacher forcing ratio for scheduling teacher forcing ratio (0 ~ 1)')
    parser.add_argument('--kl_anneal_cyclical', default=True, action='store_true', help='use cyclical mode')
    parser.add_argument('--kl_anneal_ratio', type=float, default=1, help='The decay ratio of kl annealing')
    parser.add_argument('--kl_anneal_cycle', type=int, default=4, help='The number of cycle for kl annealing during training (if use cyclical mode)')
    parser.add_argument('--seed', default=1, type=int, help='manual seed')
    parser.add_argument('--n_past', type=int, default=2, help='number of frames to condition on')
    parser.add_argument('--n_future', type=int, default=10, help='number of frames to predict')
    parser.add_argument('--n_eval', type=int, default=30, help='number of frames to predict at eval time')
    parser.add_argument('--rnn_size', type=int, default=256, help='dimensionality of hidden layer')
    parser.add_argument('--posterior_rnn_layers', type=int, default=1, help='number of layers')
    parser.add_argument('--predictor_rnn_layers', type=int, default=2, help='number of layers')
    parser.add_argument('--z_dim', type=int, default=64, help='dimensionality of z_t')
    parser.add_argument('--p_dim', type=int, default=7, help='dimensionality of pos')
    parser.add_argument('--g_dim', type=int, default=128, help='dimensionality of encoder output vector and decoder input vector')
    parser.add_argument('--beta', type=float, default=0.0001, help='weighting on KL to prior')
    parser.add_argument('--num_workers', type=int, default=4, help='number of data loading threads')
    parser.add_argument('--last_frame_skip', action='store_true', help='if true, skip connections go between frame t and frame t+t rather than last ground truth frame')
    parser.add_argument('--cuda', default=True, action='store_true')  

    args = parser.parse_args()
    return args

def train(x, cond, modules, optimizer, kl_anneal, args):
    modules['frame_predictor'].zero_grad()
    modules['posterior'].zero_grad()
    modules['encoder'].zero_grad()
    modules['decoder'].zero_grad()

    # initialize the hidden state.
    modules['frame_predictor'].hidden = modules['frame_predictor'].init_hidden()
    modules['posterior'].hidden = modules['posterior'].init_hidden()
    mse = 0
    kld = 0
    # print(x.shape,cond.shape) 
    # torch.Size([12, 30, 3, 64, 64]) torch.Size([12, 30, 7])
    # epoch, time, channel, height, width 
    use_teacher_forcing = True if random.random() < args.tfr else False
    x = x.permute(1, 0, 2, 3 ,4)
    cond = cond.permute(1, 0, 2)
    x = x.to(torch.device('cuda'))
    cond = cond.to(torch.device('cuda'))
    #print(x.shape,cond.shape) 
    pred_list = [x[0]]
    # teacher forcing
    # reconstruct loss + KL loss
    for i in range(1, args.n_past + args.n_future):
        #raise NotImplementedError
        if use_teacher_forcing:
            prev_x = x[i-1]
        else:
            prev_x = pred_list[i-1]
        
        h_t, _ = modules['encoder'](x[i])
        z_t, mu, logvar = modules['posterior'](h_t)

        if i < args.n_past:
            h_prev, skip = modules['encoder'](prev_x)
        else:
            h_prev, _= modules['encoder'](prev_x)
        # concat cond, h_prev, z_t
        # print(cond[i].shape,h_prev.shape,z_t.shape)
        g_t = modules['frame_predictor'](torch.cat([cond[i],h_prev,z_t],1))
        x_pred = modules['decoder']([g_t,skip])
        pred_list.append(x_pred)
        mse += modules['mseloss'](x_pred,x[i])
        kld += kl_criterion(mu, logvar,args)
    #print("here",len(x[args.n_past:args.n_future]),len(pred_list[args.n_past:args.n_future]))
    _,_,psnr = finn_eval_seq(x[args.n_past:args.n_future],pred_list[args.n_past:args.n_future])
    beta = kl_anneal.get_beta()
    loss = mse + kld * beta
    loss.backward()
    optimizer.step()

    return loss.detach().cpu().numpy() / (args.n_past + args.n_future), mse.detach().cpu().numpy() / (args.n_past + args.n_future), kld.detach().cpu().numpy() / (args.n_future + args.n_past)

class kl_annealing():
    def __init__(self, args):
        super().__init__()
        self.beta = 0
        self.cyclical = args.kl_anneal_cyclical
        self.ratio = args.kl_anneal_ratio
        self.cycle = args.kl_anneal_cycle
        self.iters = args.niter * args.epoch_size
        self.idx = -1

        if self.cyclical:
            iter_cycle = self.iters/self.cycle
            tmp1 = np.linspace(0, 1, num = int(iter_cycle * self.ratio))
            tmp2 = np.ones(int(iter_cycle - int(iter_cycle * self.ratio)))
            tmp3 = np.concatenate((tmp1,tmp2))
            print("cyclical",tmp1.shape,tmp2.shape)
            for i in range(self.cycle - 1):
                tmp3 = np.concatenate((tmp3,tmp1,tmp2))
        else:
            tmp1 = np.linspace(0, 1, num = int(self.iters * self.ratio))
            tmp2 = np.ones(int(self.iters - int(self.iters * self.ratio)))
            print("monotonic",tmp1.shape,tmp2.shape)
            tmp3 = np.concatenate((tmp1,tmp2))
        
        self.Betas = tmp3
        print("Beta shape",self.Betas.shape)
        plt.plot(self.Betas)
        plt.grid()
        plt.savefig("a.png")
            
    def update(self):
        self.idx = self.idx + 1

    def get_beta(self):
        self.update()
        return self.Betas[self.idx]


def main():
    args = parse_args()
    if args.cuda:
        assert torch.cuda.is_available(), 'CUDA is not available.'
        device = 'cuda'
    else:
        device = 'cpu'
    
    assert args.n_past + args.n_future <= 30 and args.n_eval <= 30
    assert 0 <= args.tfr and args.tfr <= 1
    assert 0 <= args.tfr_start_decay_epoch 
    assert 0 <= args.tfr_decay_step and args.tfr_decay_step <= 1

    if args.model_dir != '':
        # load model and continue training from checkpoint
        saved_model = torch.load('%s/model.pth' % args.model_dir)
        optimizer = args.optimizer
        model_dir = args.model_dir
        niter = args.niter
        args = saved_model['args']
        args.optimizer = optimizer
        args.model_dir = model_dir
        args.log_dir = '%s/continued' % args.log_dir
        start_epoch = saved_model['last_epoch']
    else:
        name = 'rnn_size=%d-predictor-posterior-rnn_layers=%d-%d-n_past=%d-n_future=%d-lr=%.4f-g_dim=%d-z_dim=%d-last_frame_skip=%s-beta=%.7f'\
            % (args.rnn_size, args.predictor_rnn_layers, args.posterior_rnn_layers, args.n_past, args.n_future, args.lr, args.g_dim, args.z_dim, args.last_frame_skip, args.beta)

        args.log_dir = '%s/%s' % (args.log_dir, name)
        niter = args.niter
        start_epoch = 0

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs('%s/gen/' % args.log_dir, exist_ok=True)

    print("Random Seed: ", args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if os.path.exists('./{}/train_record.txt'.format(args.log_dir)):
        os.remove('./{}/train_record.txt'.format(args.log_dir))
    
    print(args)

    with open('./{}/train_record.txt'.format(args.log_dir), 'a') as train_record:
        train_record.write('args: {}\n'.format(args))

    # ------------ build the models  --------------

    if args.model_dir != '':
        frame_predictor = saved_model['frame_predictor']
        posterior = saved_model['posterior']
    else:
        frame_predictor = lstm(args.g_dim+args.z_dim+args.p_dim, args.g_dim, args.rnn_size, args.predictor_rnn_layers, args.batch_size, device)
        print(args.g_dim+args.z_dim, args.g_dim, args.rnn_size, args.predictor_rnn_layers, args.batch_size, device)
        posterior = gaussian_lstm(args.g_dim, args.z_dim, args.rnn_size, args.posterior_rnn_layers, args.batch_size, device)
        frame_predictor.apply(init_weights)
        posterior.apply(init_weights)
            
    if args.model_dir != '':
        decoder = saved_model['decoder']
        encoder = saved_model['encoder']
        mse_criterion = saved_model['mseloss']
    else:
        encoder = vgg_encoder(args.g_dim)
        decoder = vgg_decoder(args.g_dim)
        encoder.apply(init_weights)
        decoder.apply(init_weights)
        mse_criterion = nn.MSELoss()
    
    # --------- transfer to device ------------------------------------
    frame_predictor.to(device)
    posterior.to(device)
    encoder.to(device)
    decoder.to(device)
    mse_criterion.to(device)

    # --------- load a dataset ------------------------------------
    train_data = bair_robot_pushing_dataset(args, 'train')
    validate_data = bair_robot_pushing_dataset(args, 'validate')
    train_loader = DataLoader(train_data,
                            num_workers=args.num_workers,
                            batch_size=args.batch_size,
                            shuffle=True,
                            drop_last=True,
                            pin_memory=True)
    train_iterator = iter(train_loader)

    validate_loader = DataLoader(validate_data,
                            num_workers=args.num_workers,
                            batch_size=args.batch_size,
                            shuffle=True,
                            drop_last=True,
                            pin_memory=True)

    validate_iterator = iter(validate_loader)

    # ---------------- optimizers ----------------
    if args.optimizer == 'adam':
        args.optimizer = optim.Adam
    elif args.optimizer == 'rmsprop':
        args.optimizer = optim.RMSprop
    elif args.optimizer == 'sgd':
        args.optimizer = optim.SGD
    else:
        raise ValueError('Unknown optimizer: %s' % args.optimizer)

    params = list(frame_predictor.parameters()) + list(posterior.parameters()) + list(encoder.parameters()) + list(decoder.parameters())
    optimizer = args.optimizer(params, lr=args.lr, betas=(args.beta1, 0.999))
    kl_anneal = kl_annealing(args)

    modules = {
        'frame_predictor': frame_predictor,
        'posterior': posterior,
        'encoder': encoder,
        'decoder': decoder,
        'mseloss':mse_criterion
    }
    # --------- training loop ------------------------------------

    progress = tqdm(total=args.niter)
    best_val_psnr = 0
    for epoch in range(start_epoch, start_epoch + niter):
        frame_predictor.train()
        posterior.train()
        encoder.train()
        decoder.train()

        epoch_loss = 0
        epoch_mse = 0
        epoch_kld = 0

        for _ in range(args.epoch_size):
            try:
                seq, cond = next(train_iterator)
            except StopIteration:
                train_iterator = iter(train_loader)
                seq, cond = next(train_iterator)
            
            loss, mse, kld = train(seq, cond, modules, optimizer, kl_anneal, args)
            epoch_loss += loss
            epoch_mse += mse
            epoch_kld += kld
        
        if epoch >= args.tfr_start_decay_epoch:
            ### Update teacher forcing ratio ###
            if args.tfr_decay_step == 0:
                args.tfr = max(0, 1-(epoch-args.tfr_start_decay_epoch)/(args.niter-args.tfr_start_decay_epoch))
            else:
                args.tfr = max(0, args.tfr - args.tfr_decay_step)


        progress.update(1)
        with open('./{}/train_record.txt'.format(args.log_dir), 'a') as train_record:
            train_record.write(('[epoch: %02d] loss: %.5f | mse loss: %.5f | kld loss: %.5f\n' % (epoch, epoch_loss  / args.epoch_size, epoch_mse / args.epoch_size, epoch_kld / args.epoch_size)))
        with open('./{}/KL.txt'.format(args.log_dir), 'a') as train_record:
            train_record.write(('{}\n'.format(epoch_kld / args.epoch_size)))


        frame_predictor.eval()
        encoder.eval()
        decoder.eval()
        posterior.eval()

        if epoch % 5 == 0:
            psnr_list = []
            for _ in range(len(validate_data) // args.batch_size):
                try:
                    validate_seq, validate_cond = next(validate_iterator)
                except StopIteration:
                    validate_iterator = iter(validate_loader)
                    validate_seq, validate_cond = next(validate_iterator)
                validate_seq = validate_seq.permute(1, 0, 2, 3 ,4)
                validate_cond = validate_cond.permute(1, 0, 2)
                pred_seq = pred(validate_seq, validate_cond, modules, args, device)
                plot_pred(validate_seq, pred_seq, epoch, args)
                _, _, psnr = finn_eval_seq(validate_seq[args.n_past:args.n_future], pred_seq[args.n_past:args.n_future])
                psnr_list.append(psnr)

                
            ave_psnr = np.mean(np.concatenate(psnr))


            with open('./{}/train_record.txt'.format(args.log_dir), 'a') as train_record:
                train_record.write(('====================== validate psnr = {:.5f} ========================\n'.format(ave_psnr)))
            with open('./{}/ave_psnr.txt'.format(args.log_dir), 'a') as train_record:
                train_record.write(('{:.5f}\n'.format(ave_psnr)))


            if ave_psnr > best_val_psnr:
                best_val_psnr = ave_psnr
                # save the model
                torch.save({
                    'encoder': encoder,
                    'decoder': decoder,
                    'frame_predictor': frame_predictor,
                    'posterior': posterior,
                    'args': args,
                    'mseloss':mse_criterion,
                    'last_epoch': epoch},
                    '%s/model.pth' % args.log_dir)

        if epoch % 20 == 0:
            try:
                validate_seq, validate_cond = next(validate_iterator)
            except StopIteration:
                validate_iterator = iter(validate_loader)
                validate_seq, validate_cond = next(validate_iterator)



if __name__ == '__main__':
    main()
        
