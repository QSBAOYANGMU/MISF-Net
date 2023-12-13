import torch
import os
import argparse
import numpy as np
from datasets.crowd import Crowd
from models.VPMFNet import fusion_model
from utils.evaluation import eval_game, eval_relative
from PIL import Image


parser = argparse.ArgumentParser(description='Test')
parser.add_argument('--data-dir', default='F:\CrowdCounting(rgbt)\RGBTCC-main\RGBT-CC',
                        help='training data directory')
parser.add_argument('--save-dir', default=r'F:\CrowdCounting(rgbt)\DEFNet-main\save_D\1015-190733',
                        help='model directory')
parser.add_argument('--model', default='model_last.pth'
                    , help='model name')

parser.add_argument('--device', default='0', help='gpu device')
args = parser.parse_args()

if __name__ == '__main__':


    datasets = Crowd(os.path.join(args.data_dir, 'test'), method='test')
    dataloader = torch.utils.data.DataLoader(datasets, 1, shuffle=False,
                                             num_workers=8, pin_memory=True)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.device  # set vis gpu
    device = torch.device('cuda')

    model = fusion_model()
    model.to(device)
    model_path = os.path.join(args.save_dir, args.model)
    checkpoint = torch.load(model_path, device)
    model.load_state_dict(checkpoint)
    model.eval()

    print('testing...')
    # Iterate over data.
    game = [0, 0, 0, 0]
    mse = [0, 0, 0, 0]
    total_relative_error = 0

    for inputs, target, name in dataloader:
        if type(inputs) == list:
            inputs[0] = inputs[0].to(device)
            inputs[1] = inputs[1].to(device)
        else:
            inputs = inputs.to(device)

        # inputs are images with different sizes
        if type(inputs) == list:
            assert inputs[0].size(0) == 1
        else:
            assert inputs.size(0) == 1, 'the batch size should equal to 1 in validation mode'
        with torch.set_grad_enabled(False):
            outputs= model(inputs)
            # outputs,_,_,_ = outputs
            for L in range(4):
                abs_error, square_error = eval_game(outputs, target, L)
                game[L] += abs_error
                mse[L] += square_error
            relative_error = eval_relative(outputs, target)
            total_relative_error += relative_error

    # print(type(type(outputs)))

            path = os.path.join(r'C:\Users\NBU\Desktop\人群计数2\test_dark',name[0]+'.png')
            img =outputs.data.cpu().numpy()
            img = img.squeeze(0).squeeze(0)
            print(name[0],img.sum())
            img = img * 255.
            img = Image.fromarray(img).convert('L')
            img.save(path)


    N = len(dataloader)
    game = [m / N for m in game]
    mse = [torch.sqrt(m / N) for m in mse]
    total_relative_error = total_relative_error / N

    log_str = 'Test{}, GAME0 {game0:.2f} GAME1 {game1:.2f} GAME2 {game2:.2f} GAME3 {game3:.2f} ' \
              'MSE {mse:.2f} Re {relative:.4f}, '.\
        format(N, game0=game[0], game1=game[1], game2=game[2], game3=game[3], mse=mse[0], relative=total_relative_error)

    print(log_str)
