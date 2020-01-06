from __future__ import unicode_literals
import random
import numpy as np
from collections import OrderedDict
import torch

# from datasets.pascalvoc import PascalVOC
from datasets.corrosion import Corrosion
# import generators.deeplabv2 as deeplabv2
import generators.unet as unet
import discriminators.discriminator_unet as dis
from torchvision import transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
from utils.transforms import RandomSizedCrop, IgnoreLabelClass, ToTensorLabel, NormalizeOwn,ZeroPadding, OneHotEncode, RandomSizedCrop3
from utils.lr_scheduling import poly_lr_scheduler
import torch.nn.functional as F
import torch.nn as nn
from functools import reduce
import torch.optim as optim
import os
import cv2
import argparse
from torchvision.transforms import ToTensor,Compose
from utils.validate import val, val_e
from utils.helpers import pascal_palette_invert
import torchvision.transforms as transforms
import PIL.Image as Image
# from discriminators.discriminator_unet import Dis
import torch.utils.model_zoo as model_zoo
from torchsummary import summary
from utils.loss import CrossEntropy2d
from utils.roi_crf import img_crf, get_roi_crf

# from utils.log import setup_logging, ResultsLog, save_checkpoint, export_args_namespace

home_dir = os.path.dirname(os.path.realpath(__file__))
colnames = ['epoch', 'iter', 'LD', 'LD_fake', 'LD_real', 'LG', 'LG_ce', 'LG_adv', 'LG_semi']

def parse_args():

    # Parse arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("prefix",
                        help="Prefix to identify current experiment")

    parser.add_argument("dataset_dir", default='/media/data/seg_dataset', 
                        help="A directory containing img (Images) and cls (GT Segmentation) folder")

    parser.add_argument("--mode", choices=('base','label_correction'),default='base',
                        help="base (baseline),label_correction")

    parser.add_argument("--lam_adv",default=0.5,
                        help="Weight for Adversarial loss for Segmentation Network training")

    parser.add_argument("--lam_semi",default=0.1,
                        help="Weight for Semi-supervised loss")

    parser.add_argument("--t_semi",default=0.2,type=float,
                        help="Threshold for self-taught learning")

    parser.add_argument("--nogpu",action='store_true',
                        help="Train only on cpus. Helpful for debugging")

    parser.add_argument("--max_epoch",default=600,type=int,
                        help="Maximum iterations.")

    parser.add_argument("--start_epoch",default=1,type=int,
                        help="Resume training from this epoch")

    parser.add_argument("--snapshot", default='snapshots',
                        help="Snapshot to resume training")

    parser.add_argument("--snapshot_dir",default=os.path.join(home_dir,'data','snapshots'),
                        help="Location to store the snapshot")

    parser.add_argument("--batch_size",default=8,type=int, # 10
                        help="Batch size for training")

    parser.add_argument("--val_orig",action='store_true',
                        help="Do Inference on original size image. Otherwise, crop to 320x320 like in training ")

    parser.add_argument("--d_label_smooth",default=0.1,type=float,
                        help="Label smoothing for real images in Discriminator")

    parser.add_argument("--d_optim",choices=('sgd','adam'),default='sgd',
                        help="Discriminator Optimizer.")

    parser.add_argument("--no_norm",action='store_true',
                        help="No Normalizaion on the Images")

    parser.add_argument("--init_net",choices=('imagenet','mscoco', 'unet'),default='mscoco',
                        help="Pretrained Net for Segmentation Network")

    parser.add_argument("--d_lr",default=0.00005,type=float, # 0.00005
                        help="lr for discriminator")

    parser.add_argument("--g_lr",default=1e-4,type=float, # 0.00025
                        help="lr for generator")

    parser.add_argument("--seed",default=1,type=int,
                        help="Seed for random numbers used in semi-supervised training")

    parser.add_argument("--wait_semi",default=100,type=int,
                        help="Number of Epochs to wait before using semi-supervised loss")

    parser.add_argument("--split",default=1,type=float) # 0.8
    # args = parser.parse_args()

    parser.add_argument("--dtrain_times",default=1,type=int)
    args = parser.parse_args()

    return args

'''
    Snapshot the Best Model
'''
def snapshot(model,valoader,epoch,best_miou,snapshot_dir,prefix):
    miou = val(model,valoader)
    # eiou = val_sigmoid(model,valoader)
    snapshot = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'miou': miou
    }
    if miou > best_miou:
        best_miou = miou
        torch.save(snapshot,os.path.join(snapshot_dir,'{}.pth.tar'.format(prefix)))

    print("[{}] Curr mIoU: {:0.4f} Best mIoU: {}".format(epoch,miou,best_miou))
    return best_miou

'''
    Snapshot the Best Model
'''
def snapshote(model,valoader,epoch,best_miou,best_eiou,snapshot_dir,prefix):
    miou, eiou = val_e(model,valoader)
    # eiou = val_sigmoid(model,valoader)
    # eiou = -2
    snapshot = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'miou': miou,
        'eiou': eiou
    }
    if miou > best_miou:
        best_miou = miou
        torch.save(snapshot,os.path.join(snapshot_dir,'{}_lcorrection_miou.pth.tar'.format(prefix)))

    if eiou > best_eiou:
        best_eiou = eiou
        torch.save(snapshot,os.path.join(snapshot_dir,'{}_lcorrection_eiou.pth.tar'.format(prefix)))

    print("[{}] Curr mIoU: {:0.4f} Curr eIoU: {:0.4f} Best mIoU: {:0.4f} Best eIoU: {:0.4f}".format(epoch,miou,eiou,best_miou,best_eiou))
    return best_miou, best_eiou

def make_D_label(label, ignore_mask):
    ignore_mask = np.expand_dims(ignore_mask, axis=1)
    # print("ignore_mask: ", ignore_mask.shape)
    D_label = np.ones(ignore_mask.shape)*label
    # print("D_label: ", D_label.shape)
    D_label[ignore_mask] = 255 # 255
    # print("D_label: ", D_label.shape)
    D_label = Variable(torch.FloatTensor(D_label)).cuda(args.gpu)
    # print("D_label: ", D_label.size())
    return D_label


def loss_calc(pred, label):
    """
    This function returns cross entropy loss for semantic segmentation
    """
    # out shape batch_size x channels x h x w -> batch_size x channels x h x w
    # label shape h x w x 1 x batch_size  -> batch_size x 1 x h x w
    label = Variable(label.long()).cuda()
    criterion = CrossEntropy2d().cuda()
    return criterion(pred, label)


'''
    Use PreTrained Model for Initial Weights
'''
def init_weights(model,init_net):
    if init_net == 'imagenet':
        # Pretrain on ImageNet
        inet_weights = model_zoo.load_url('https://download.pytorch.org/models/resnet101-5d3b4d8f.pth')
        del inet_weights['fc.weight']
        del inet_weights['fc.bias']
        state = model.state_dict()
        state.update(inet_weights)
        model.load_state_dict(state)
    elif init_net == 'mscoco':
        # TODO: Upload the weights somewhere to use load.url()
        filename = os.path.join(home_dir,'data','MS_DeepLab_resnet_pretrained_COCO_init.pth')
        assert(os.path.isfile(filename))
        saved_net = torch.load(filename)
        new_state = model.state_dict()
        saved_net = {k.partition('Scale.')[2]: v for i, (k,v) in enumerate(saved_net.items())}
        new_state.update(saved_net)
        model.load_state_dict(new_state)
    elif init_net == 'unet':
        unet.init_weights(model, init_type='normal')

def iter_test(trainloader_u):
    loader = iter(trainloader_u)
    cnt = 0
    while(True):
        try:
            cnt += 1
            img, mask, ohmask = next(loader)
            print("{}: {}".format(cnt, img.size()))
        except:
            print("next failed!!")
            break

'''
    Baseline Training
'''
def train_base(generator,optimG,trainloader,valoader,args):

    best_miou = -1
    best_eiou = -1
    for epoch in range(args.start_epoch,args.max_epoch+1):
        generator.train()
        for batch_id, (img, mask, _, _, _) in enumerate(trainloader):

            if args.nogpu:
                img,mask = Variable(img),Variable(mask)
            else:
                img,mask = Variable(img.cuda()),Variable(mask.cuda())

            itr = len(trainloader)*(epoch-1) + batch_id
            cprob = generator(img)
            cprob = nn.LogSoftmax()(cprob)
            # cprob = nn.Sigmoid()(cprob)

            Lseg = nn.NLLLoss()(cprob,mask)
            # Lseg = nn.BCELoss()(cprob,mask)

            optimG = poly_lr_scheduler(optimG, args.g_lr, itr)
            optimG.zero_grad()

            Lseg.backward()
            optimG.step()

            # print("[{}][{}]Loss: {:0.4f}".format(epoch,itr,Lseg.data[0]))
            print("[{}][{}]Loss: {:0.4f}".format(epoch,itr,Lseg.data))

        best_miou, best_eiou = snapshote(generator,valoader,epoch,best_miou,best_eiou,args.snapshot_dir,args.prefix)


def train_label_correction(generator,optimG,trainloader,valoader,args):
    best_miou = -1
    best_eiou = -1
    # FILE_INFO = file_info()

    for epoch in range(args.start_epoch,args.max_epoch+1):
        generator.train()
        for batch_id, (img_l, mask_l, ohmask_l, _, imgs_org) in enumerate(trainloader):

            itr = len(trainloader)*(epoch-1) + batch_id
            if args.nogpu:
                img_l,mask_l,ohmask_l = Variable(img_l),Variable(mask_l),Variable(ohmask_l)
            else:
                img_l,mask_l,ohmask_l = Variable(img_l.cuda()),Variable(mask_l.cuda()),Variable(ohmask_l.cuda())
            imgs_array = imgs_org.numpy()

            LGcorr = torch.tensor(0)
            loss_corr_value = 0
            # begin label correction

            ################################################
            #  train labelled data                         #
            ################################################
            cpmap = generator(img_l)
            cpmapsmax = nn.Softmax2d()(cpmap)
            cpmaplsmax = nn.LogSoftmax()(cpmap)
            LGce = nn.NLLLoss()(cpmaplsmax,mask_l)
            LGce_d = LGce.data
            optimG = poly_lr_scheduler(optimG, args.g_lr, itr)
            optimG.zero_grad()
            LGce.backward()
            optimG.step()

            ################################################
            #  label correction                            #
            ################################################
            if epoch > args.wait_semi:
                # Check if the loader has a batch available

                # semi unlabelled training
                cpmap = generator(img_l)
                cpmapsmax = nn.Softmax2d()(cpmap)
                cpmaplsmax = nn.LogSoftmax()(cpmap)

                soft_preds_np = cpmapsmax.data.cpu().numpy()
                soft_mask = np.zeros((soft_preds_np.shape[0],
                                    soft_preds_np.shape[2], soft_preds_np.shape[3]))
                for i in range(soft_preds_np.shape[0]):
                    soft_pred = soft_preds_np[i]
                    img_array = imgs_array[i]
                    img_array = cv2.resize(img_array, (320,320), interpolation = cv2.INTER_AREA) 
                    hard_pred = np.argmax(soft_pred,axis=0).astype(np.uint8)
                    rois_crf, rois = get_roi_crf(img_array, soft_pred, hard_pred)
                    for roi_crf, roi in zip(rois_crf, rois):
                        # print(roi)
                        xmin, ymin, xmax, ymax = roi[0], roi[1], roi[2], roi[3]
                        soft_pred[:, ymin:ymax, xmin:xmax] = roi_crf
                    crf_result = np.argmax(soft_pred,axis=0).astype(np.uint8)
                    soft_mask[i] = crf_result
                mask_c = Variable(torch.tensor(soft_mask)).type(torch.LongTensor).cuda()
                LGcorr = nn.NLLLoss()(cpmaplsmax,mask_c)
                optimG = poly_lr_scheduler(optimG, args.g_lr, itr)
                optimG.zero_grad()
                LGcorr.backward()
                optimG.step()

            LGcorr_d = LGcorr.data
            LGseg_d = LGce_d + args.lam_adv*LGcorr
            print("[{}][{}] LGseg_d: {:.4f} LGce_d: {:.4f} LG_corr: {:.4f}"\
                    .format(epoch,itr,LGseg_d,LGce_d,LGcorr_d))

        # best_miou = snapshot(generator,valoader,epoch,best_miou,args.snapshot_dir,args.prefix)
        best_miou, best_eiou = snapshote(generator,valoader,epoch,best_miou,best_eiou,args.snapshot_dir,args.prefix)

def main():

    args = parse_args()

    random.seed(0)
    torch.manual_seed(0)
    if not args.nogpu:
        torch.cuda.manual_seed_all(0)

    if args.no_norm:
        imgtr = [ToTensor()]
    else:
        imgtr = [ToTensor(),NormalizeOwn()]

    # softmax
    labtr = [IgnoreLabelClass(),ToTensorLabel()]
    # cotr = [RandomSizedCrop((320,320))] # (321,321)
    cotr = [RandomSizedCrop3((320,320))]

    print("dataset_dir: ", args.dataset_dir)
    trainset_l = Corrosion(home_dir,args.dataset_dir,img_transform=Compose(imgtr), 
                           label_transform=Compose(labtr),co_transform=Compose(cotr),
                           split=args.split,labeled=True, label_correction=True)
    trainloader_l = DataLoader(trainset_l,batch_size=args.batch_size,shuffle=True,
                               num_workers=2,drop_last=True)

    #########################
    # Validation Dataloader #
    ########################
    if args.val_orig:
        if args.no_norm:
            imgtr = [ZeroPadding(),ToTensor()]
        else:
            imgtr = [ZeroPadding(),ToTensor(),NormalizeOwn()]
        # softmax
        labtr = [IgnoreLabelClass(),ToTensorLabel()]
        cotr = []
    else:
        if args.no_norm:
            imgtr = [ToTensor()]
        else:
            imgtr = [ToTensor(),NormalizeOwn()]
        # softmax
        labtr = [IgnoreLabelClass(),ToTensorLabel()]
        cotr = [RandomSizedCrop3((320,320))]

    valset = Corrosion(home_dir,args.dataset_dir,img_transform=Compose(imgtr), \
        label_transform = Compose(labtr),co_transform=Compose(cotr),train_phase=False)
    valoader = DataLoader(valset,batch_size=1)

    #############
    # GENERATOR #
    #############
    # generator = deeplabv2.ResDeeplab()

    # softmax generator: in_chs=3, out_chs=2
    generator = unet.AttU_Net()
    init_weights(generator,args.init_net)

    if args.init_net != 'unet':
        """
        optimG = optim.SGD(filter(lambda p: p.requires_grad, \
            generator.parameters()),lr=args.g_lr,momentum=0.9,\
            weight_decay=0.0001,nesterov=True)
        """
        optimG = optim.Adam(filter(lambda p: p.requires_grad, \
            generator.parameters()),args.g_lr, [0.5, 0.999])
    else:
        optimG = optim.Adam(filter(lambda p: p.requires_grad, \
            generator.parameters()),args.g_lr, [0.5, 0.999])

    if not args.nogpu:
        generator = nn.DataParallel(generator).cuda()

    if args.mode == 'base':
        train_base(generator,optimG,trainloader_l,valoader,args)
    elif args.mode == 'label_correction':
        print("mode: ", args.mode)
        # train_semi(generator,discriminator,optimG,optimD,trainloader_l,trainloader_u,valoader,args)
        train_label_correction(generator,optimG,trainloader_l,valoader,args)
    else:
        # train_semir(generator,discriminator,optimG,optimD,trainloader_l,valoader,args)
        print("training mode incorrect")

if __name__ == '__main__':
    main()
