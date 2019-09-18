import argparse
import datetime
import os
from math import log

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
from tqdm import tqdm

from model import Glow

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description='Glow trainer')
parser.add_argument('--batch', default=32, type=int, help='batch size')
parser.add_argument('--resume', default=False, type=bool, help='resume training')
parser.add_argument('--resume_exp', default=None, type=str, help='resume experiment log dir')
parser.add_argument('--iter', default=200000, type=int, help='maximum iterations')
parser.add_argument(
    '--n_flow', default=32, type=int, help='number of flows in each block'
)
parser.add_argument('--n_block', default=4, type=int, help='number of blocks')
parser.add_argument(
    '--no_lu',
    action='store_true',
    help='use plain convolution instead of LU decomposed version',
)
parser.add_argument(
    '--affine', action='store_true', help='use affine coupling instead of additive'
)
parser.add_argument('--n_bits', default=5, type=int, help='number of bits')
parser.add_argument('--lr', default=1e-5, type=float, help='learning rate')
parser.add_argument('--warm', default=False, type=bool, help="whether or not warmup learning rate")
parser.add_argument('--img_size', default=64, type=int, help='image size')
parser.add_argument('--temp', default=0.7, type=float, help='temperature of sampling')
parser.add_argument('--n_sample', default=32, type=int, help='number of samples')
parser.add_argument('--sample_freq', default=100, type=int, help='interval of sample and reverse')
parser.add_argument('--check_freq', default=1000, type=int, help='interval of save checkpoints')
parser.add_argument('--experiment_dir', default='experiment', type=str,
                    help="experiments directory save the samples and checkpoint")
parser.add_argument('path', metavar='PATH', type=str, required=True, help='Path to image(dataset) directory')


def sample_data(path, batch_size, image_size):
    transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            # transforms.CenterCrop(image_size),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (1, 1, 1)),
        ]
    )

    dataset = datasets.ImageFolder(path, transform=transform)
    loader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=4)
    loader = iter(loader)

    while True:
        try:
            yield next(loader)

        except StopIteration:
            loader = DataLoader(
                dataset, shuffle=True, batch_size=batch_size, num_workers=4
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


def train(args, model, optimizer):
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
        for key, value in vars(args).items():
            f.write(key+"\t"+value+'\n')

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
            image, _ = next(dataset)
            image = image.to(device)

            if i == 0:
                with torch.no_grad():
                    log_p, logdet, _ = model.module(image + torch.rand_like(image) / n_bins)

                    continue

            else:
                log_p, logdet, z_encode = model(image + torch.rand_like(image) / n_bins)

            logdet = logdet.mean()

            loss, log_p, log_det = calc_loss(log_p, logdet, args.img_size, n_bins)
            model.zero_grad()
            loss.backward()

            if args.warm:
                warmup_lr = args.lr * min(1, i * args.batch_size / (50000 * 10))
            else:
                warmup_lr = args.lr
            optimizer.param_groups[0]['lr'] = warmup_lr

            optimizer.step()

            pbar.set_description(
                f'Loss: {loss.item():.5f}; logP: {log_p.item():.5f}; logdet: {log_det.item():.5f}; lr: {warmup_lr:.7f}'
            )

            if i % args.sample_freq == 0:
                with torch.no_grad():
                    z_sample = []
                    for z in z_shapes:
                        z_new = torch.randn(args.n_sample, *z) * args.temp
                        z_sample.append(z_new.to(device))

                    # sample
                    utils.save_image(
                        model_single.reverse(z_sample).cpu().data,
                        os.path.join(log_dir, 'sample', f'{str(i).zfill(6)}.png'),
                        normalize=True,
                        nrow=args.n_sample//4,
                        range=(-0.5, 0.5),
                    )
                    # sample at first
                    utils.save_image(
                        model_single.reverse(z_sample_first).cpu().data,
                        os.path.join(log_dir, 'sample', f'{str(i).zfill(6)}_first.png'),
                        normalize=True,
                        nrow=args.n_sample//4,
                        range=(-0.5, 0.5),
                    )
                    # reconstruct
                    utils.save_image(
                        model_single.reverse(z_encode).cpu().data,
                        os.path.join(log_dir, 'sample', f'{str(i).zfill(6)}_reverse.png'),
                        normalize=True,
                        nrow=args.batch_size//4,
                        range=(-0.5, 0.5),
                    )

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

    model_single = Glow(
        3, args.n_flow, args.n_block, affine=args.affine, conv_lu=not args.no_lu
    )
    model = nn.DataParallel(model_single)
    # model = model_single
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train(args, model, optimizer)
