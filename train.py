import argparse
import datetime
import os
from math import log

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import utils
from tqdm import tqdm

from data.explo import ExploDataset
from model import Glow, Pix2PixGlow, Encoder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description='Glow trainer')
parser.add_argument('--model', default='pix2pixglow', type=str, choices=['glow', 'pix2pixglow'],
                    help='which type of model')
parser.add_argument('--batch', default=32, type=int, help='batch size')
parser.add_argument('--resume', default=False, type=bool, help='resume training')
parser.add_argument('--resume_exp', default=None, type=str, help='resume experiment log dir')
parser.add_argument('--iter', default=200000, type=int, help='maximum iterations')
parser.add_argument('--n_flow', default=32, type=int, help='number of flows in each block')
parser.add_argument('--n_block', default=4, type=int, help='number of blocks')
parser.add_argument('--no_lu', action='store_true',
                    help='use plain convolution instead of LU decomposed version',)
parser.add_argument('--affine', action='store_true', help='use affine coupling instead of additive')
parser.add_argument('--n_bits', default=5, type=int, help='number of bits')
parser.add_argument('--lr', default=1e-5, type=float, help='learning rate')
parser.add_argument('--warm', default=False, type=bool, help="whether or not warmup learning rate")
parser.add_argument('--img_size', default=64, type=int, help='image size')
parser.add_argument('--temp', default=0.7, type=float, help='temperature of sampling')
parser.add_argument('--n_sample', default=32, type=int, help='number of samples')
parser.add_argument('--sample_freq', default=500, type=int, help='interval of sample and reverse')
parser.add_argument('--check_freq', default=2000, type=int, help='interval of save checkpoints')
parser.add_argument('--experiment_dir', default='experiment', type=str,
                    help="experiments directory save the samples and checkpoint")
parser.add_argument('path', metavar='PATH', type=str, help='Path to image(dataset) directory')


def sample_data(path, batch_size, image_size):
    dataset = ExploDataset(path)
    loader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=batch_size//2)
    loader = iter(loader)

    while True:
        try:
            yield next(loader)

        except StopIteration:
            loader = DataLoader(
                dataset, shuffle=True, batch_size=batch_size, num_workers=batch_size//2
            )
            loader = iter(loader)
            yield next(loader)


def calc_z_shapes(n_channel, input_size, n_flow, n_block):
    z_shapes = []

    for i in range(n_block - 1):
        input_size //= 2
        n_channel *= 2

        z_shapes.append((n_channel, input_size, input_size))

    input_size //= 2
    z_shapes.append((n_channel * 4, input_size, input_size))

    return z_shapes


def calc_loss(log_p, logdet, image_size, n_bins):
    # log_p = calc_log_p([z_list])
    n_pixel = image_size * image_size * 3

    loss = -log(n_bins) * n_pixel
    loss = loss + logdet + log_p

    return (
        (-loss / (log(2) * n_pixel)).mean(),
        (log_p / (log(2) * n_pixel)).mean(),
        (logdet / (log(2) * n_pixel)).mean(),
    )


def calc_loss_triple(log_p_c, logdet_c, z_c,
                     log_p_t, logdet_t, z_t,
                     z_s, z_criterion, n_z,
                     image_size, n_bins):
    n_pixel = image_size * image_size * 3

    loss = -log(n_bins) * n_pixel
    loss = loss + logdet_c + log_p_c + logdet_t + log_p_t
    z_loss = 0.0
    # last n_z z s to calculate loss
    # currently, only support 1
    assert n_z == 1, NotImplementedError
    z_cs = z_c[-1] + z_s
    z_loss = z_loss + z_criterion(z_cs, z_t[-1])
    loss = -loss / (log(2) * n_pixel) + 0.1 * z_loss

    return (
        loss.mean(),
        z_loss.mean(),
        (log_p_c / (log(2) * n_pixel)).mean(),
        (logdet_c / (log(2) * n_pixel)).mean(),
        (log_p_t / (log(2) * n_pixel)).mean(),
        (logdet_t / (log(2) * n_pixel)).mean(),
    )


def train(args, model, encoder, optimizer, z_criterion):
    # setup log dir
    date = str(datetime.datetime.now())
    date = date[:date.rfind(":")].replace("-", "").replace(":", "").replace(" ", "_")
    log_dir = os.path.join(args.experiment_dir, "exp_"+date)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(os.path.join(log_dir, "sample")):
        os.makedirs(os.path.join(log_dir, "sample"))
    if not os.path.exists(os.path.join(log_dir, "checkpoint")):
        os.makedirs(os.path.join(log_dir, "checkpoint"))

    # dump args
    with open(os.path.join(log_dir, 'opts.txt'), 'w') as f:
        f.write("experiment_log_dir: "+str(log_dir)+'\n')
        for key, value in vars(args).items():
            f.write(str(key)+": "+str(value)+'\n')

    # open loss log file
    loss_log = open(os.path.join(log_dir, 'losses.txt'), 'w')

    dataset = iter(sample_data(args.path, args.batch, args.img_size))
    n_bins = 2. ** args.n_bits

    # resume training
    if args.resume and args.resume_exp is not None:
        assert os.path.exists(os.path.join(args.experiment_dir, args.resume_exp)), args.resume_exp + " does not exist"
        latest_model = os.path.join(args.experiment_dir, args.resume_exp, 'checkpoint', 'model_latest.pt')
        latest_optimizer = os.path.join(args.experiment_dir, args.resume_exp, 'checkpoint', 'optimizer_latest.pt')
        model = model.load_state_dict(torch.load(latest_model))
        optimizer = optimizer.load_state_dict(torch.load(latest_optimizer))

    # calculate the z shapes
    z_shapes = calc_z_shapes(3, args.img_size, args.n_flow, args.n_block)

    # sample the first z which is consistent during traing, used to measure the performance
    z_sample_first = []
    for z in z_shapes:
        z_new = torch.randn(args.n_sample, *z) * args.temp
        z_sample_first.append(z_new.to(device))

    with tqdm(range(args.iter)) as pbar:
        for i in pbar:
            data = next(dataset)
            base = data['base'].to(device)
            image = data['image'].to(device)
            # font = data['font'].to(device)
            # char = data['char'].to(device)
            attr = data['attr'].to(device)
            noise = torch.randn((image.size(0), 8)).to(device)
            attr = torch.cat([attr, noise], dim=1)

            if i == 0:
                with torch.no_grad():
                    # log_p, logdet, _ = model.module(image + torch.rand_like(image) / n_bins)
                    model.module(base, image)
                    encoder.module(attr)

                    continue

            else:
                # log_p size: [batch_size]
                # logdet size: [gpu_num]
                # z_encode: list, len = n_block, refer to cal_z_shapes
                # log_p, logdet, z_encode = model(image + torch.rand_like(image) / n_bins)
                log_p_c, logdet_c, z_encode_c, log_p_t, logdet_t, z_encode_t = model(base, image)
                z_s = encoder(attr)

            logdet_c = logdet_c.mean()
            logdet_t = logdet_t.mean()

            # loss, log_p, log_det = calc_loss(log_p, logdet, args.img_size, n_bins)
            loss, z_loss, log_p_c, log_det_c, log_p_t, log_det_t = \
                calc_loss_triple(log_p_c, logdet_c, z_encode_c,
                                 log_p_t, logdet_t, z_encode_t,
                                 z_s, z_criterion, 1,
                                 args.img_size, n_bins)
            model.zero_grad()
            loss.backward()

            if args.warm:
                warmup_lr = args.lr * min(1, i * args.batch / (50000 * 10))
            else:
                warmup_lr = args.lr
            optimizer.param_groups[0]['lr'] = warmup_lr

            optimizer.step()

            # log loss
            message = (
                f'Loss: {loss.item():.5f}; '
                f'logP_c: {log_p_c.item():.5f}; logdet_c: {log_det_c.item():.5f}; '
                f'logP_t: {log_p_t.item():.5f}; logdet_t: {log_det_t.item():.5f}; '
                f'lr: {warmup_lr:.7f}'
            )
            pbar.set_description(message)
            log_message = f'Step: {i}/{args.iter}; ' + message
            loss_log.write(log_message+'\n')
            loss_log.flush()

            if i % args.sample_freq == 0:
                with torch.no_grad():
                    z_sample = []
                    for z in z_shapes:
                        z_new = torch.randn(args.n_sample, *z) * args.temp
                        z_sample.append(z_new.to(device))

                    # sample at first
                    utils.save_image(
                        model_single.reverse(z_sample_first, z_sample_first)[1].cpu().data,
                        os.path.join(log_dir, 'sample', f'{str(i).zfill(6)}_first.png'),
                        normalize=True,
                        nrow=args.n_sample//4,
                        range=(-0.5, 0.5),
                    )
                    # sample
                    utils.save_image(
                        model_single.reverse(z_sample, z_sample)[1].cpu().data,
                        os.path.join(log_dir, 'sample', f'{str(i).zfill(6)}_sample.png'),
                        normalize=True,
                        nrow=args.n_sample//4,
                        range=(-0.5, 0.5),
                    )
                    # reverse
                    utils.save_image(
                        model_single.reverse(z_sample, z_sample, reconstruct=True)[1].cpu().data,
                        os.path.join(log_dir, 'sample', f'{str(i).zfill(6)}_sample_reconstruct.png'),
                        normalize=True,
                        nrow=args.batch//4,
                        range=(-0.5, 0.5),
                    )
                    # reconstruct
                    utils.save_image(
                        model_single.reverse(z_encode_c, z_encode_t, reconstruct=True)[1].cpu().data,
                        os.path.join(log_dir, 'sample', f'{str(i).zfill(6)}_reconstruct.png'),
                        normalize=True,
                        nrow=args.batch//4,
                        range=(-0.5, 0.5),
                    )
                    # ground truth
                    utils.save_image(
                        image.cpu().data,
                        os.path.join(log_dir, 'sample', f'{str(i).zfill(6)}_ground-truth.png'),
                        normalize=True,
                        nrow=args.batch//4,
                        range=(-0.5, 0.5),
                    )

            # save checkpoint
            if i % args.check_freq == 0:
                torch.save(
                    model.state_dict(), os.path.join(log_dir, 'checkpoint', f'model_{str(i).zfill(6)}.pt')
                )
                torch.save(
                    model.state_dict(), os.path.join(log_dir, 'checkpoint', 'model_latest.pt')
                )
                torch.save(
                    optimizer.state_dict(), os.path.join(log_dir, 'checkpoint', f'optimizer_{str(i).zfill(6)}.pt')
                )
                torch.save(
                    optimizer.state_dict(), os.path.join(log_dir, 'checkpoint', 'optimizer_latest.pt')
                )


if __name__ == '__main__':
    args = parser.parse_args()
    print(args)

    if args.model == 'pix2pixglow':
        model_single = Pix2PixGlow(
            3, args.n_flow, args.n_block, affine=args.affine, conv_lu=not args.no_lu
        )

    elif args.model == 'glow':
        model_single = Glow(
            3, args.n_flow, args.n_block, affine=args.affine, conv_lu=not args.no_lu
        )
    else:
        raise NotImplementedError

    model = nn.DataParallel(model_single)
    # model = model_single
    model = model.to(device)

    encoder_single = Encoder(37)
    encoder = nn.DataParallel(encoder_single)
    encoder = encoder.to(device)

    all_parameters = list(model.parameters()) + list(encoder.parameters())
    optimizer = optim.Adam(all_parameters, lr=args.lr)

    z_criterion = nn.MSELoss()
    train(args, model, encoder, optimizer, z_criterion)
