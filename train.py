from utils.regression_trainer import RegTrainer
import argparse
import os
import torch
args = None
import warnings
warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description='Train ')
    parser.add_argument("--use_cuda", type=bool, default=True)
    parser.add_argument("--seed", type=int, default=2022)
    parser.add_argument('--data-dir', default='F:\CrowdCounting(rgbt)\RGBTCC-main\RGBT-CC_600',
                        help='training data directory')
    parser.add_argument('--save-dir', default='F:/CrowdCounting(rgbt)/DEFNet-main/save',
                        help='directory to save models.')
    parser.add_argument('--lr', type=float, default=1e-5,#1e-5
                        help='the initial learning rate')
    parser.add_argument('--resume', default=r'',##F:\CrowdCounting(rgbt)\DEFNet-main\0628-153606_10.92/best_model_25.pth
                        help='the path of resume training model')
    parser.add_argument('--device', default='0', help='assign device')
    parser.add_argument('--crop-size', type=int, default=256,
                        help='defacult 256')

    # default
    parser.add_argument('--weight-decay', type=float, default=1e-4,##1e-4
                        help='the weight decay')
    parser.add_argument('--max-model-num', type=int, default=1,
                        help='max models num to save ')
    parser.add_argument('--max-epoch', type=int, default=500,
                        help='max training epoch')
    parser.add_argument('--val-epoch', type=int, default=1,
                        help='the num of steps to log training information')
    parser.add_argument('--val-start', type=int, default=20,
                        help='the epoch start to val')
    parser.add_argument('--save-all-best', type=bool, default=True,
                        help='whether to load opt state')
    parser.add_argument('--batch-size', type=int, default=2,
                        help='train batch size')
    parser.add_argument('--num-workers', type=int, default=2,
                        help='the num of training process')
    parser.add_argument('--downsample_ratio', type=int, default=8,##8
                        help='downsample ratio')
    parser.add_argument('--use-background', type=bool, default=True,
                        help='whether to use background modelling')
    parser.add_argument('--sigma', type=float, default=8.0,
                        help='sigma for likelihood')
    parser.add_argument('--background-ratio', type=float, default=0.15,
                        help='background ratio')
    parser.add_argument('--a', type=float, default=0.0001,#0.0001#0.5
                        help='weight_1')
    parser.add_argument('--b', type=float, default=0.00001,#1e-05##0.5
                        help='weight_2')
    parser.add_argument('--c', type=float, default=0,
                        help='weight_3')
    # parser.add_argument('--m', type=float, default=1,
    #                     help='weight_main')
    # parser.add_argument('--bia', type=float, default=1e-5,
    #                     help='bias')
    # parser.add_argument("--decay_interval", type=int, default=1000)
    # parser.add_argument("--decay_ratio", type=float, default=0.5)
    # parser.add_argument('--lr_scheduler', default="StepLR", type=str, help='CosineAnnealingLR or StepLR')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    torch.backends.cudnn.benchmark = True
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device.strip()  # set vis gpu
    trainer = RegTrainer(args)
    trainer.setup()
    trainer.train()
