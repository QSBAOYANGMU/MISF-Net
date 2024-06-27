from utils.evaluation import eval_game, eval_relative
from utils.trainer import Trainer
from utils.helper import Save_Handle, AverageMeter
from utils.functions import MSE
from utils.functions import CMD
from utils.functions import DiffLoss
from utils.functions import PatchEmbed
from torch.optim import lr_scheduler
import random
import pytorch_ssim
from torchvision.utils import make_grid
import   scipy.io  as  sio
import os
import sys
import time
import torch
from tensorboardX import SummaryWriter
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
import logging
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from datasets.crowd import Crowd
from losses.bay_loss import Bay_Loss
from losses.post_prob import Post_Prob
from models.DEFNet import fusion_model
import warnings
warnings.filterwarnings("ignore")
def train_collate(batch):
    transposed_batch = list(zip(*batch))
    if type(transposed_batch[0][0]) == list:
        rgb_list = [item[0] for item in transposed_batch[0]]
        t_list = [item[1] for item in transposed_batch[0]]
        rgb = torch.stack(rgb_list, 0)
        t = torch.stack(t_list, 0)
        images = [rgb, t]
    else:
        images = torch.stack(transposed_batch[0], 0)
    points = transposed_batch[1]  # the number of points is not fixed, keep it as a list of tensor
    st_sizes = torch.FloatTensor(transposed_batch[2])
    return images, points, st_sizes


class RegTrainer(Trainer):
    def setup(self):
        """initial the datasets, model, loss and optimizer"""
        args = self.args
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        # torch.cuda.manual_seed_all(args.seed) # 让显卡产生的随机数一致
        np.random.seed(args.seed)  # numpy产生的随机数一致
        # random.seed(args.seed)
        # args = self.args
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_count = torch.cuda.device_count()
            # for code conciseness, we release the single gpu version
            assert self.device_count == 1
            logging.info('using {} gpus'.format(self.device_count))
        else:
            raise Exception("gpu is not available")

        self.downsample_ratio = args.downsample_ratio
        self.datasets = {x: Crowd(os.path.join(args.data_dir, x),
                                  args.crop_size,
                                  args.downsample_ratio,
                                  x) for x in ['train', 'val', 'test']}
        self.dataloaders = {x: DataLoader(self.datasets[x],
                                          collate_fn=(train_collate
                                                      if x == 'train' else default_collate),
                                          batch_size=(args.batch_size
                                                      if x == 'train' else 1),
                                          shuffle=(True if x == 'train' else False),
                                          num_workers=args.num_workers * self.device_count,
                                          pin_memory=(True if x == 'train' else False))
                            for x in ['train', 'val', 'test']}

        self.model = fusion_model()
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.writer = SummaryWriter('../result_tensorboard')####
        self.start_epoch = 0
        if args.resume:
            suf = args.resume.rsplit('.', 1)[-1]
            if suf == 'tar':
                checkpoint = torch.load(args.resume, self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.start_epoch = checkpoint['epoch'] + 1
            elif suf == 'pth':
                self.model.load_state_dict(torch.load(args.resume, self.device))

        self.post_prob = Post_Prob(args.sigma,
                                   args.crop_size,
                                   args.downsample_ratio,
                                   args.background_ratio,
                                   args.use_background,
                                   self.device)
        self.criterion = Bay_Loss(args.use_background, self.device)
        self.mse = torch.nn.MSELoss(size_average=None, reduce=None, reduction='mean')
        # self.diffloss = pytorch_ssim.SSIM(window_size = 11)
        self.diffloss = torch.nn.MSELoss(size_average=None, reduce=None, reduction='mean')
        # self.diffloss = torch.nn.CosineSimilarity(dim=2, eps=1e-6)
        # self.diffloss=torch.nn.CosineEmbeddingLoss(margin=0.5, size_average=None, reduce=None, reduction='mean')
        self.cmd = torch.nn.MSELoss(size_average=None, reduce=None, reduction='mean')
        # self.cmd=pytorch_ssim.SSIM(window_size = 11)
        # if torch.cuda.is_available() and args.use_cuda:
        self.mse = self.mse.to(self.device)
        self.diffloss = self.diffloss.to(self.device)
        self.cmd = self.cmd.to(self.device)

        self.save_list = Save_Handle(max_num=args.max_model_num)
        # self.lr_scheduler_name = args.lr_scheduler
        self.best_game0 = np.inf
        self.best_game3 = np.inf
        self.best_count = 0
        self.best_count_1 = 0
        # if self.lr_scheduler_name == "StepLR":
        #     self.scheduler = lr_scheduler.StepLR(self.optimizer,
        #                                          last_epoch=-1,
        #                                          step_size=args.decay_interval,
        #                                          gamma=args.decay_ratio)
        # elif self.lr_scheduler_name == "CosineAnnealingLR":
        #     self.scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer,
        #                                                     T_max=self.max_epochs * self.num_steps_per_epoch,
        #                                                     last_epoch=-1)
        # else:
        #     raise Exception("Wrong lr_scheduler_name")

    def train(self):
        """training process"""
        args = self.args
        for epoch in range(self.start_epoch, args.max_epoch):
            # logging.info('save dir: '+args.save_dir)
            logging.info('-'*5 + 'Epoch {}/{}'.format(epoch, args.max_epoch - 1) + '-'*5)
            self.epoch = epoch
            # self.current_epoch=self.epoch
            # self.image_path=r'F:\CrowdCounting(rgbt)\DEFNet-main\feature map'
            self.train_eopch()
            # a, b, c,d = self.train_eopch()
            # self.writer.add_scalar('loss_tatal', a, global_step=self.epoch)
            # self.writer.add_scalar('loss_similarity', b, global_step=self.epoch)
            # self.writer.add_scalar('loss_difference', c, global_step=self.epoch)
            # self.writer.add_scalar('loss_reconstruction', d, global_step=self.epoch)
            # sio.savemat(os.path.join(self.image_path,'1s.mat'), mdict = {'L': d,})
            # sio.savemat(os.path.join(self.image_path,'1ts.mat'), mdict={'L': e, })
            # sio.savemat(os.path.join(self.image_path,'1.mat'), mdict={'L': f, })
            # sio.savemat(os.path.join(self.image_path,'1t.mat'), mdict={'L': g, })
            # epoch=20
            if epoch % args.val_epoch == 0 and epoch >= args.val_start:
                game0_is_best, game3_is_best = self.val_epoch()

            if epoch >= args.val_start and (game0_is_best or game3_is_best):
                self.test_epoch()

    def train_eopch(self):
        args = self.args
        # self.current_epoch = epoch
        epoch_loss = AverageMeter()
        epoch_loss_CMD = AverageMeter()
        epoch_loss_DIFF = AverageMeter()
        epoch_loss_MSE = AverageMeter()

        epoch_game = AverageMeter()
        epoch_mse = AverageMeter()
        epoch_start = time.time()
        self.model.train()  # Set model to training mode

        # Iterate over data.
        for step, (inputs, points, st_sizes) in enumerate(self.dataloaders['train']):

            if type(inputs) == list:
                inputs[0] = inputs[0].to(self.device)
                inputs[1] = inputs[1].to(self.device)
            else:
                inputs = inputs.to(self.device)
            st_sizes = st_sizes.to(self.device)
            gd_count = np.array([len(p) for p in points], dtype=np.float32)
            points = [p.to(self.device) for p in points]

            with torch.set_grad_enabled(True):
                outputs,  in_data_16_shared,in_data_8_shared,\
               in_data_4_shared,in_data_2_shared,in_data_1_shared,\
               in_data_16t_shared,in_data_8t_shared,in_data_4t_shared,\
               in_data_2t_shared,in_data_1t_shared,\
               in_data_16t_private,in_data_16_private,\
               in_data_8t_private,in_data_8_private,in_data_4t_private,in_data_4_private,in_data_2t_private,in_data_2_private, \
               in_data_1, in_data_1_d, in_data_2, in_data_2_d, in_data_4, in_data_4_d, \
               in_data_8, in_data_8_d, in_data_16, in_data_16_d,\
               in_data_1t_private,in_data_1_private = self.model(inputs)
                prob_list = self.post_prob(points, st_sizes)
                loss_bays = self.criterion(prob_list, outputs)
                #####similarity#######
                # loss_CMD = self.cmd(in_data_16_shared, in_data_16_t_shared, 5)
                loss_CMD = self.cmd(in_data_16_shared, in_data_16t_shared)
                loss_CMD+=self.cmd(in_data_8_shared, in_data_8t_shared)
                loss_CMD+=self.cmd(in_data_4_shared, in_data_4t_shared)
                loss_CMD+=self.cmd(in_data_2_shared, in_data_2t_shared)
                loss_CMD+=self.cmd(in_data_1_shared, in_data_1t_shared)
                # Between private and shared####diff loss
                # Tar = torch.tensor([-1]).to(self.device)
                # batch_size=args.batch_size
                # loss_DIFF = self.diffloss(in_data_16_shared.reshape(batch_size, -1), in_data_16_private.reshape(batch_size, -1),Tar)#+self.diffloss(in_data_16_t_shared, in_data_16_t_private)
                # loss_DIFF = self.diffloss(in_data_16_shared.reshape(batch_size, -1), in_data_16_private.reshape(batch_size, -1),Tar)#+self.diffloss(in_data_16_t_shared, in_data_16_t_private)
                # loss_DIFF = self.diffloss(in_data_16t_private, in_data_16_private)
                loss_DIFF = torch.log10(1 / torch.sqrt(self.diffloss(in_data_16t_private, in_data_16_private)))
                # print(loss_DIFF)
                # loss_DIFF += self.diffloss(in_data_16_t_shared.reshape(batch_size, -1), in_data_16_t_private.reshape(batch_size, -1),Tar)
                # loss_DIFF += self.diffloss(in_data_8t_private, in_data_8_private)
                loss_DIFF += torch.log10(1 / torch.sqrt(self.diffloss(in_data_8t_private, in_data_8_private)))
                # among private
                # loss_DIFF += self.diffloss(in_data_16_private.reshape(batch_size, -1), in_data_16_t_private.reshape(batch_size, -1),Tar)
                # loss_DIFF += self.diffloss(in_data_4t_private, in_data_4_private)
                loss_DIFF += torch.log10(1 / torch.sqrt(self.diffloss(in_data_4t_private, in_data_4_private)))
                # loss_DIFF += self.diffloss(in_data_2t_private, in_data_2_private)
                loss_DIFF += torch.log10(1 / torch.sqrt(self.diffloss(in_data_2t_private, in_data_2_private)))
                # loss_DIFF += self.diffloss(in_data_1t_private, in_data_1_private)
                loss_DIFF += torch.log10(1 / torch.sqrt(self.diffloss(in_data_1t_private, in_data_1_private)))
                ####
                loss_MSE = self.mse(in_data_1t_private + in_data_1t_shared, in_data_1_d)
                loss_MSE += self.mse(in_data_2t_private + in_data_2t_shared, in_data_2_d)
                loss_MSE += self.mse(in_data_4t_private + in_data_4t_shared, in_data_4_d)
                loss_MSE += self.mse(in_data_8t_private + in_data_8t_shared, in_data_8_d)
                loss_MSE += self.mse(in_data_16t_private + in_data_16t_shared, in_data_16_d)
                loss_MSE += self.mse(in_data_16_private + in_data_16_shared, in_data_16)
                loss_MSE += self.mse(in_data_8_private + in_data_8_shared, in_data_8)
                loss_MSE += self.mse(in_data_4_private + in_data_4_shared, in_data_4)
                loss_MSE += self.mse(in_data_2_private + in_data_2_shared, in_data_2)
                loss_MSE += self.mse(in_data_1_private + in_data_1_shared, in_data_1)

                #####
                loss=loss_bays+args.a*loss_CMD+args.b*loss_DIFF+args.c*loss_MSE
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if type(inputs) == list:
                    N = inputs[0].size(0)
                else:
                    N = inputs.size(0)
                pre_count = torch.sum(outputs.view(N, -1), dim=1).detach().cpu().numpy()
                # lr = self.optimizer.param_groups[0]['lr']
                res = pre_count - gd_count
                epoch_loss.update(loss.item(), N)
                epoch_loss_CMD.update(loss_CMD.item(), N)
                epoch_loss_DIFF.update(loss_DIFF.item(), N)
                epoch_loss_MSE.update(loss_MSE.item(), N)

                epoch_mse.update(np.mean(res * res), N)
                epoch_game.update(np.mean(abs(res)), N)

        logging.info('Epoch {} Train, Loss: {:.7f},loss_CMD: {:.7f},loss_DIFF: {:.7f},loss_MSE: {:.2f}, GAME0: {:.2f} MSE: {:.2f}, Cost {:.1f} sec'
                     .format(self.epoch, epoch_loss.get_avg(),epoch_loss_CMD.get_avg(),epoch_loss_DIFF.get_avg(), epoch_loss_MSE.get_avg(), epoch_game.get_avg(), np.sqrt(epoch_mse.get_avg()),
                             time.time()-epoch_start))
        # tensorboard_ind += 1
        model_state_dic = self.model.state_dict()
        save_path = os.path.join(self.save_dir, '{}_ckpt.tar'.format(self.epoch))
        torch.save({
            'epoch': self.epoch,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'model_state_dict': model_state_dic
        }, save_path)
        self.save_list.append(save_path)  # control the number of saved models
        # if self.lr_scheduler_name == "StepLR":
        #     self.scheduler.step()
        return epoch_loss.get_avg(),epoch_loss_CMD.get_avg(),epoch_loss_DIFF.get_avg(),epoch_loss_MSE.get_avg()#feature_map_s1,feature_map_s1t,feature_map_1,feature_map_1d
    def val_epoch(self):
        args = self.args
        self.model.eval()  # Set model to evaluate mode

        # Iterate over data.
        game = [0, 0, 0, 0]
        mse = [0, 0, 0, 0]
        total_relative_error = 0

        for inputs, target, name in self.dataloaders['val']:
            if type(inputs) == list:
                inputs[0] = inputs[0].to(self.device)
                inputs[1] = inputs[1].to(self.device)
            else:
                inputs = inputs.to(self.device)

            # inputs are images with different sizes
            if type(inputs) == list:
                assert inputs[0].size(0) == 1
            else:
                assert inputs.size(0) == 1, 'the batch size should equal to 1 in validation mode'
            with torch.set_grad_enabled(False):
                outputs, in_data_16_shared,in_data_8_shared,\
               in_data_4_shared,in_data_2_shared,in_data_1_shared,\
               in_data_16t_shared,in_data_8t_shared,in_data_4t_shared,\
               in_data_2t_shared,in_data_1t_shared,\
               in_data_16t_private,in_data_16_private,\
               in_data_8t_private,in_data_8_private,in_data_4t_private,in_data_4_private,in_data_2t_private,in_data_2_private, \
               in_data_1, in_data_1_d, in_data_2, in_data_2_d, in_data_4, in_data_4_d, \
               in_data_8, in_data_8_d, in_data_16, in_data_16_d,\
               in_data_1t_private,in_data_1_private= self.model(inputs)
                #outputs,_,_,_ = outputs
                for L in range(4):
                    abs_error, square_error = eval_game(outputs, target, L)
                    game[L] += abs_error
                    mse[L] += square_error
                relative_error = eval_relative(outputs, target)
                total_relative_error += relative_error

        N = len(self.dataloaders['val'])
        game = [m / N for m in game]
        mse = [torch.sqrt(m / N) for m in mse]
        total_relative_error = total_relative_error / N

        logging.info('Epoch {} Val{}, '
                     'GAME0 {game0:.2f} GAME1 {game1:.2f} GAME2 {game2:.2f} GAME3 {game3:.2f} MSE {mse:.2f} Re {relative:.4f}, '
                     .format(self.epoch, N, game0=game[0], game1=game[1], game2=game[2], game3=game[3], mse=mse[0], relative=total_relative_error
                             )
                     )

        model_state_dic = self.model.state_dict()

        game0_is_best = game[0] < self.best_game0
        game3_is_best = game[3] < self.best_game3

        if game[0] < self.best_game0 or game[3] < self.best_game3:
            self.best_game3 = min(game[3], self.best_game3)
            self.best_game0 = min(game[0], self.best_game0)
            logging.info("*** Best Val GAME0 {:.3f} GAME3 {:.3f} model epoch {}".format(self.best_game0,
                                                                                    self.best_game3,
                                                                                    self.epoch))
            if args.save_all_best:
                torch.save(model_state_dic, os.path.join(self.save_dir, 'best_model_{}.pth'.format(self.best_count)))
                self.best_count += 1
            else:
                torch.save(model_state_dic, os.path.join(self.save_dir, 'best_model.pth'))

        return game0_is_best, game3_is_best

    def test_epoch(self):
        self.model.eval()  # Set model to evaluate mode

        # Iterate over data.
        game = [0, 0, 0, 0]
        mse = [0, 0, 0, 0]
        total_relative_error = 0

        for inputs, target, name in self.dataloaders['test']:
            if type(inputs) == list:
                inputs[0] = inputs[0].to(self.device)
                inputs[1] = inputs[1].to(self.device)
            else:
                inputs = inputs.to(self.device)

            # inputs are images with different sizes
            if type(inputs) == list:
                assert inputs[0].size(0) == 1
            else:
                assert inputs.size(0) == 1, 'the batch size should equal to 1 in validation mode'
            with torch.set_grad_enabled(False):
                outputs, in_data_16_shared,in_data_8_shared,\
               in_data_4_shared,in_data_2_shared,in_data_1_shared,\
               in_data_16t_shared,in_data_8t_shared,in_data_4t_shared,\
               in_data_2t_shared,in_data_1t_shared,\
               in_data_16t_private,in_data_16_private,\
               in_data_8t_private,in_data_8_private,in_data_4t_private,in_data_4_private,in_data_2t_private,in_data_2_private, \
               in_data_1, in_data_1_d, in_data_2, in_data_2_d, in_data_4, in_data_4_d, \
               in_data_8, in_data_8_d, in_data_16, in_data_16_d,\
               in_data_1t_private,in_data_1_private= self.model(inputs)
                # outputs,_,_,_, = outputs
                for L in range(4):
                    abs_error, square_error = eval_game(outputs, target, L)
                    game[L] += abs_error
                    mse[L] += square_error
                relative_error = eval_relative(outputs, target)
                total_relative_error += relative_error

        N = len(self.dataloaders['test'])
        game = [m / N for m in game]
        mse = [torch.sqrt(m / N) for m in mse]
        total_relative_error = total_relative_error / N

        logging.info('Epoch {} Test{}, '
                     'GAME0 {game0:.2f} GAME1 {game1:.2f} GAME2 {game2:.2f} GAME3 {game3:.2f} MSE {mse:.2f} Re {relative:.4f}, '
                     .format(self.epoch, N, game0=game[0], game1=game[1], game2=game[2], game3=game[3], mse=mse[0],
                             relative=total_relative_error
                             )
                     )



