# -*- coding: UTF-8 -*-
import argparse
import os
import torch


def build_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument('--traindir', type=str, default='../train/labeled_data/')
    parser.add_argument('--train_metadir', type=str, default='../', help='metadir/metafile, 存储对应image label pairs的路径.')
    parser.add_argument('--train_metafile', type=str, default='train_label.csv', help='csv文件名')

    parser.add_argument('--testdir', type=str, default='../test')
    parser.add_argument('--test_metadir', type=str, default=None)
    parser.add_argument('--test_metafile', type=str, default=None)

    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--step_size', type=int, default=5)
    parser.add_argument('--gamma', type=int, default=0.9)
    parser.add_argument('--ckpt_name', type=str, help='用于保存checkpoints的名称')
    parser.add_argument('--submission_name', type=str, default='submission.csv', help='存submit的名字')
    parser.add_argument('--model_name', type=str, help='本次训练用到的model_type')
    parser.add_argument('--kfold', default=5, help='交叉验证Fold_num')
    opt = parser.parse_args(args=[])
    
    opt.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    opt.resultdir = './result'
    if not os.path.exists(opt.resultdir):
        os.makedirs(opt.resultdir)

    print('.' * 75)
    for key in opt.__dict__:
        param = opt.__dict__[key]
        print('...param: {}: {}'.format(key, param))
    print('.' * 75)

    return opt


