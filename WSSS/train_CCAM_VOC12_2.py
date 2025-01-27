# Copyright (C) 2020 * Ltd. All rights reserved.
# author : Sanghyeon Jo <josanghyeokn@gmail.com>
# modified by Sierkinhane <sierkinhane@163.com>

import sys

import matplotlib

matplotlib.use('Agg')
from torchvision import transforms
from tensorboardX import SummaryWriter

from torch.utils.data import DataLoader
from WSSS.datamodules.fgbg_datamodule import ForegroundTextureDataModule
from WSSS.utils import *
from WSSS.core.datasets import *
from WSSS.core.model import *
from WSSS.tools.general.io_utils import *
from WSSS.tools.general.time_utils import *
from WSSS.tools.general.json_utils import *
from WSSS.core.loss import *

from WSSS.tools.ai.log_utils import *
from WSSS.tools.ai.demo_utils import *
from WSSS.tools.ai.torch_utils import *
from WSSS.tools.ai.evaluate_utils import *

from WSSS.tools.ai.augment_utils import *
from WSSS.tools.ai.randaugment import *
from shutil import copyfile
import matplotlib.pyplot as plt
from WSSS.optimizer import PolyOptimizer

os.environ["NUMEXPR_NUM_THREADS"] = "8"
parser = argparse.ArgumentParser()

###############################################################################
# Dataset
###############################################################################
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--num_workers', default=8, type=int)
parser.add_argument('--data_dir', default='/data1/xjheng/dataset/VOC2012/', type=str)

###############################################################################
# Network
###############################################################################
parser.add_argument('--architecture', default='resnet50', type=str)
parser.add_argument('--mode', default='normal', type=str)  # fix

###############################################################################
# Hyperparameter
###############################################################################
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--max_epoch', default=10, type=int)
parser.add_argument('--depth', default=50, type=int)

parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--wd', default=1e-4, type=float)
parser.add_argument('--nesterov', default=True, type=str2bool)

parser.add_argument('--image_size', default=448, type=int)
parser.add_argument('--min_image_size', default=320, type=int)
parser.add_argument('--max_image_size', default=640, type=int)

parser.add_argument('--print_ratio', default=0.2, type=float)

parser.add_argument('--tag', default='', type=str)
parser.add_argument('--augment', default='', type=str)

parser.add_argument('--alpha', type=float, default=0.25)
parser.add_argument('--pretrained', type=str, required=True,
                    help='adopt different pretrained parameters, [supervised, mocov2, detco]')

flag = True

if __name__ == '__main__':
    # global flag
    ###################################################################################
    # Arguments
    ###################################################################################
    args = parser.parse_args()

    log_dir = create_directory('./experiments/logs/')
    data_dir = create_directory('./experiments/data/')
    model_dir = create_directory('./experiments/models/')
    tensorboard_dir = create_directory('./experiments/tensorboards/{}/'.format(args.tag))

    log_path = log_dir + '{}.txt'.format(args.tag)
    data_path = data_dir + '{}.json'.format(args.tag)
    model_path = model_dir + '{}.pth'.format(args.tag)
    cam_path = './experiments/images/{}'.format(args.tag)
    create_directory(cam_path)
    create_directory(cam_path + '/train')
    create_directory(cam_path + '/test')
    create_directory(cam_path + '/train/colormaps')
    create_directory(cam_path + '/test/colormaps')

    set_seed(args.seed)
    log_func = lambda string='': log_print(string, log_path)

    log_func('[i] {}'.format(args.tag))
    log_func()

    ###################################################################################
    # Transform, Dataset, DataLoader
    ###################################################################################

    # test_transform = transforms.Compose([
    #     Top_Left_Crop_For_Segmentation(args.image_size),
    # ])
    test_transform = None
    # data augmentation
    train_transform = transforms.Compose([
        transforms.Resize(size=(244, 244)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=(195, 195)),
    ])

    module = ForegroundTextureDataModule(transforms=train_transform, dataset_type='FashionMNIST')
    train_loader, _, _ = module.return_dataloaders()

    module = ForegroundTextureDataModule(transforms=test_transform, dataset_type='FashionMNIST')
    _, train_loader_for_seg, valid_loader_for_seg = module.return_dataloaders()

    log_func('[i] mean values is {}'.format(imagenet_mean))
    log_func('[i] std values is {}'.format(imagenet_std))
    log_func('[i] train_transform is {}'.format(train_transform))
    log_func('[i] test_transform is {}'.format(test_transform))
    log_func('[i] #train data'.format(len(train_loader)))
    log_func('[i] #valid data'.format(len(valid_loader_for_seg)))
    log_func()

    val_iteration = len(train_loader)
    log_iteration = int(val_iteration * args.print_ratio)
    max_iteration = args.max_epoch * val_iteration

    log_func('[i] log_iteration : {:,}'.format(log_iteration))
    log_func('[i] val_iteration : {:,}'.format(val_iteration))
    log_func('[i] max_iteration : {:,}'.format(max_iteration))

    ###################################################################################
    # Network
    ###################################################################################
    model = get_model(pretrained=args.pretrained)
    param_groups = model.get_parameter_groups()

    model = model.cuda()
    model.train()
    # model_info(model)

    log_func('[i] Architecture is {}'.format(args.architecture))
    log_func('[i] Total Params: %.2fM' % (calculate_parameters(model)))
    log_func()

    try:
        use_gpu = os.environ['CUDA_VISIBLE_DEVICES']
    except KeyError:
        use_gpu = '0'

    the_number_of_gpu = len(use_gpu.split(','))
    if the_number_of_gpu > 1:
        log_func('[i] the number of gpu : {}'.format(the_number_of_gpu))
        model = nn.DataParallel(model)

    load_model_fn = lambda: load_model(model, model_path, parallel=the_number_of_gpu > 1)
    # save_model_fn = lambda: save_model(model, model_path, parallel=the_number_of_gpu > 1)

    ###################################################################################
    # Loss, Optimizer
    ###################################################################################
    criterion = [SimMaxLoss(metric='cos', alpha=args.alpha).cuda(), SimMinLoss(metric='cos').cuda(),
                 SimMaxLoss(metric='cos', alpha=args.alpha).cuda()]

    optimizer = PolyOptimizer([
        {'params': param_groups[0], 'lr': args.lr, 'weight_decay': args.wd},
        {'params': param_groups[1], 'lr': 2 * args.lr, 'weight_decay': 0},
        {'params': param_groups[2], 'lr': 10 * args.lr, 'weight_decay': args.wd},
        {'params': param_groups[3], 'lr': 20 * args.lr, 'weight_decay': 0},
    ], lr=args.lr, momentum=0.9, weight_decay=args.wd, max_step=max_iteration)

    #################################################################################################
    # Train
    #################################################################################################
    data_dic = {
        'train': [],
        'validation': []
    }

    train_timer = Timer()
    eval_timer = Timer()

    train_meter = Average_Meter(['loss', 'positive_loss', 'negative_loss'])

    writer = SummaryWriter(tensorboard_dir)

    for epoch in range(args.max_epoch):
        for iteration, (images, bg_images, masks,
                        bg_bg_labels, bg_fg_labels,
                        labels) in enumerate(train_loader):

            images, labels = images.cuda(), labels.cuda()

            optimizer.zero_grad()
            fg_feats, bg_feats, ccam = model(images)

            loss1 = criterion[0](fg_feats)
            loss2 = criterion[1](bg_feats, fg_feats)
            loss3 = criterion[2](bg_feats)

            loss = loss1 + loss2 + loss3

            loss.backward()
            optimizer.step()

            if epoch == 0 and iteration == (len(train_loader) - 1):
                flag = check_positive(ccam)
                print(f"Is Negative: {flag}")
            if flag:
                ccam = 1 - ccam

            train_meter.add({
                'loss': loss.item(),
                'positive_loss': loss1.item() + loss3.item(),
                'negative_loss': loss2.item(),
            })

            #################################################################################################
            # For Log
            #################################################################################################

            if (iteration + 1) % 100 == 0:
                visualize_heatmap(args.tag, images.clone().detach(), ccam, 0, iteration)
                loss, positive_loss, negative_loss = train_meter.get(clear=True)
                learning_rate = float(get_learning_rate_from_optimizer(optimizer))

                data = {
                    'epoch': epoch,
                    'max_epoch': args.max_epoch,
                    'iteration': iteration + 1,
                    'learning_rate': learning_rate,
                    'loss': loss,
                    'positive_loss': loss1,
                    'negative_loss': loss2,

                    'time': train_timer.tok(clear=True),
                }
                data_dic['train'].append(data)

                log_func('[i]\t'
                         'Epoch[{epoch:,}/{max_epoch:,}],\t'
                         'iteration={iteration:,}, \t'
                         'learning_rate={learning_rate:.4f}, \t'
                         'loss={loss:.4f}, \t'
                         'positive_loss={positive_loss:.4f}, \t'
                         'negative_loss={negative_loss:.4f}, \t'
                         'time={time:.0f}sec'.format(**data)
                         )

                writer.add_scalar('Train/loss', loss, iteration)
                writer.add_scalar('Train/learning_rate', learning_rate, iteration)
                # break
        #################################################################################################
        # Evaluation
        #################################################################################################
        # save_model_fn()
        torch.save({'state_dict': model.module.state_dict() if (the_number_of_gpu > 1) else model.state_dict(),
                    'flag': flag}, model_path)

        log_func('[i] save model')

    print(args.tag)
