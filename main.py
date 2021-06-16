from functools import partial

import torch.nn as nn
from fastai.basic_train import Learner, torch
from fastai.train import ShowGraph
from fastai.data_block import DataBunch
from torch import optim

from dataset.fracnet_dataset import FracNetTrainDataset
from dataset import transforms as tsfm
from utils.metrics import dice, recall, precision, fbeta_score
from model.unet import UNet
from model.losses import MixLoss, DiceLoss


def main(args):
    train_image_dir = args.train_image_dir
    train_label_dir = args.train_label_dir
    val_image_dir = args.val_image_dir
    val_label_dir = args.val_label_dir

    batch_size = 4
    num_workers = 4
    optimizer = optim.SGD
    criterion = MixLoss(nn.BCEWithLogitsLoss(), 0.5, DiceLoss(), 1)

    thresh = 0.1
    recall_partial = partial(recall, thresh=thresh)
    precision_partial = partial(precision, thresh=thresh)
    fbeta_score_partial = partial(fbeta_score, thresh=thresh)

    model = UNet(1, 1, first_out_channels=16)#生成网络模型
    model.load_state_dict(torch.load('./model_weights.pth'))

    model = nn.DataParallel(model.cuda()) #在GPU中加速


    transforms = [
        tsfm.Window(-200, 1000),
        tsfm.MinMaxNorm(-200, 1000)
    ]
    #预处理的用于训练的图片ds_train
    ds_train = FracNetTrainDataset(train_image_dir, train_label_dir,
        transforms=transforms)
    #数据载入
    dl_train = FracNetTrainDataset.get_dataloader(ds_train, batch_size, False,
        num_workers)
    ds_val = FracNetTrainDataset(val_image_dir, val_label_dir,
        transforms=transforms)
    dl_val = FracNetTrainDataset.get_dataloader(ds_val, batch_size, False,
        num_workers)

    #DataBunch是fastai中读取数据最基本的类，其针对不同的任务将数据集处理成合适的形式，以便送入learner进行训练
    databunch = DataBunch(dl_train, dl_val,
        collate_fn=FracNetTrainDataset.collate_fn)

    learn = Learner(
        databunch,
        model,
        opt_func=optimizer,
        loss_func=criterion,
        metrics=[dice, recall_partial, precision_partial, fbeta_score_partial]
    )

    learn.fit_one_cycle(
        5,
        1e-1,
        pct_start=0,
        div_factor=1000,
        callbacks=[
            ShowGraph(learn),
        ]
    )
    torch.save(model.module.state_dict(), "./model_weights.pth")




if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_image_dir", required=True,default ="../data/ribfrac-train-images",
        help="The training image nii directory.")
    parser.add_argument("--train_label_dir", required=True,default ="../data/ribfrac-train-labels",
        help="The training label nii directory.")
    parser.add_argument("--val_image_dir", required=True,default ="../data/ribfrac-val-images",
        help="The validation image nii directory.")
    parser.add_argument("--val_label_dir", required=True,default ="../data/ribfrac-val-labels",
        help="The validation label nii directory.")
    parser.add_argument("--save_model", default=True,
        help="Whether to save the trained model.")
    args = parser.parse_args()


    main(args)
