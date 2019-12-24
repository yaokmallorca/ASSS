from __future__ import unicode_literals
import random
import numpy as np
from collections import OrderedDict
import torch
import yaml

# from datasets.pascalvoc import PascalVOC
from datasets.corrosion import Corrosion
# import generators.deeplabv2 as deeplabv2
import generators.unet as unet
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
import os.path as osp
import argparse
from torchvision.transforms import ToTensor,Compose
from utils.validate import val, val_e
from utils.helpers import pascal_palette_invert
import torchvision.transforms as transforms
import PIL.Image as Image
from discriminators.discriminator_unet import DisSoftmax, DisSigmoid
import torch.utils.model_zoo as model_zoo
from torchsummary import summary
from utils.loss import CrossEntropy2d

# crf postprocessing
from utils.crf import DenseCRF
from addict import Dict

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

    parser.add_argument("--mode", choices=('base','adv','semi', 'joint'),default='base',
                        help="base (baseline),adv (adversarial), semi (semi-supervised)")

    parser.add_argument("--lam_adv",default=0.5,
                        help="Weight for Adversarial loss for Segmentation Network training")

    parser.add_argument("--lam_semi",default=0.1,
                        help="Weight for Semi-supervised loss")

    parser.add_argument("--lam_joint",default=1.5,
                        help="Weight for Joint training loss")

    parser.add_argument("--t_semi",default=0.5,type=float,
                        help="Threshold for self-taught learning")

    parser.add_argument("--nogpu",action='store_true',
                        help="Train only on cpus. Helpful for debugging")

    parser.add_argument("--max_epoch",default=800,type=int,
                        help="Maximum iterations.")

    parser.add_argument("--start_epoch",default=1,type=int,
                        help="Resume training from this epoch")

    parser.add_argument("--snapshot", default='snapshots',
                        help="Snapshot to resume training")

    parser.add_argument("--snapshot_dir",default=os.path.join(home_dir,'data','snapshots'),
                        help="Location to store the snapshot")

    parser.add_argument("--batch_size",default=4,type=int, # 10
                        help="Batch size for training")

    parser.add_argument("--val_orig",action='store_true',
                        help="Do Inference on original size image. Otherwise, crop to 320x320 like in training ")

    parser.add_argument("--d_label_smooth",default=0.,type=float, # 0.1
                        help="Label smoothing for real images in Discriminator")

    parser.add_argument("--d_optim",choices=('sgd','adam'),default='adam', # sgd
                        help="Discriminator Optimizer.")

    parser.add_argument("--no_norm",action='store_true',
                        help="No Normalizaion on the Images")

    parser.add_argument("--init_net",choices=('imagenet','mscoco', 'unet'),default='mscoco',
                        help="Pretrained Net for Segmentation Network")

    parser.add_argument("--d_lr",default=1e-2,type=float, # 0.00005
                        help="lr for discriminator")

    parser.add_argument("--g_lr",default=1e-4,type=float, # 0.00025
                        help="lr for generator")

    parser.add_argument("--seed",default=1,type=int,
                        help="Seed for random numbers used in semi-supervised training")

    parser.add_argument("--wait_semi",default=100,type=int,
                        help="Number of Epochs to wait before using semi-supervised loss")

    parser.add_argument("--split",default=1.,type=float) # 0.8
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
        torch.save(snapshot,os.path.join(snapshot_dir,'{}_maxmiou.pth.tar'.format(prefix)))

    if eiou > best_eiou:
        best_eiou = eiou
        torch.save(snapshot,os.path.join(snapshot_dir,'{}_maxeiou.pth.tar'.format(prefix)))

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

'''
    Snapshot the Best Model
'''
def snapshot_segdis(model,discriminator,valoader,epoch,best_miou,best_eiou,snapshot_dir,prefix):
    # miou = val(model,valoader)
    miou, eiou = val_e(model,valoader)
    snapshot = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'miou': miou
    }
    snapshot_dis = {
        'epoch': epoch,
        'state_dict': discriminator.state_dict(),
        'miou': miou
    }
    if miou > best_miou:
        best_miou = miou
        torch.save(snapshot,os.path.join(snapshot_dir,'{}_maxmiou.pth.tar'.format(prefix)))
        torch.save(snapshot_dis,os.path.join(snapshot_dir,'{}_dis_maxmiou.pth.tar'.format(prefix)))

    if eiou > best_eiou:
        best_eiou = eiou
        torch.save(snapshot,os.path.join(snapshot_dir,'{}_maxeiou.pth.tar'.format(prefix)))

    print("[{}] Curr mIoU: {:0.4f} Curr eIoU: {:0.4f} Best mIoU: {:0.4f} Best eIoU: {:0.4f}".format(epoch,miou,eiou,best_miou,best_eiou))
    return best_miou, best_eiou


def make_D_label(label, ignore_mask):
    ignore_mask = np.expand_dims(ignore_mask, axis=1)
    # print("ignore_mask: ", ignore_mask.shape)
    D_label = np.ones(ignore_mask.shape)*label
    # print("D_label: ", D_label.shape)
    D_label[ignore_mask] = 0 # 255
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

'''
    Semi supervised training
'''
def train_semi(generator,discriminator,optimG,optimD,trainloader_l,trainloader_u,valoader,args):

    best_miou = -1
    best_eiou = -1
    for epoch in range(args.start_epoch,args.max_epoch+1):
        generator.train()

        trainloader_l_iter = iter(trainloader_l)
        trainloader_u_iter = iter(trainloader_u)
        print("Epoch: {}".format(epoch))
        batch_id = 0
        while(True):
            batch_id += 1
            itr = (len(trainloader_u) + len(trainloader_l))*(epoch-1) + batch_id
            optimG.zero_grad()
            optimD.zero_grad()
            optimD = poly_lr_scheduler(optimD, args.d_lr, itr)
            optimG = poly_lr_scheduler(optimG, args.g_lr, itr)

            LGsemi_d = 0
            if epoch > args.wait_semi:
                if random.random() <0.5:
                    loader_l = trainloader_l_iter
                    loader_u = trainloader_u_iter
                else:
                    loader_l = trainloader_u_iter
                    loader_u = trainloader_l_iter
                # Check if the loader has a batch available
                try:
                    img_u,mask_u,ohmask_u,_ = next(loader_u)
                except:
                    trainloader_u_iter = iter(trainloader_u)
                    loader_u = trainloader_u_iter
                    img_u,mask_u,ohmask_u,_ = next(loader_u)

                if args.nogpu:
                    img_u,mask_u,ohmask_u = Variable(img_u),Variable(mask_u),Variable(ohmask_u)
                else:
                    img_u,mask_u,ohmask_u = Variable(img_u.cuda()),Variable(mask_u.cuda()),Variable(ohmask_u.cuda())

                # semi unlabelled training
                cpmap = generator(img_u)
                cpmapsmax = nn.Softmax2d()(cpmap)
                # confsmax = nn.Softmax2d()(conf)
                conf = discriminator(cpmapsmax)
                confsigmoid = nn.Sigmoid()(conf)
                confsigmoid_remain = confsigmoid.detach().squeeze(1)
                # conflsmax = nn.LogSoftmax()(conf)

                N = cpmap.size()[0]
                H = cpmap.size()[2]
                W = cpmap.size()[3]

                semi_gt = np.int8((confsigmoid_remain.cpu().numpy() >= args.t_semi))
                # semi_gt = cpmap.data.cpu().numpy().argmax(axis=1)
                # semi_gt[semi_ignore_mask] = 1
                semi_ratio = 1.0 - float(semi_gt.sum())/semi_gt.size
                # print('semi ratio: {:.4f}'.format(semi_ratio))
                if semi_ratio == 0.0:
                    loss_semi_value += 0
                else:
                    semi_gt = torch.FloatTensor(semi_gt)
                    LGsemi = args.lam_semi * loss_calc(cpmapsmax, semi_gt)
                    LGsemi_d += LGsemi.data.cpu().numpy()/args.lam_semi
                    # LGsemi += LGadv_d
                    LGsemi.backward()

            ################################################
            #  train labelled data                         #
            ################################################
            loader_l = trainloader_l_iter
            loader_u = trainloader_u_iter
            try:
                if random.random() <0.5:
                    img_l,mask_l,ohmask_l,_ = next(loader_l)
                else:
                    img_l,mask_l,ohmask_l,_ = next(loader_u)
            except:
                break
            if args.nogpu:
                img_l,mask_l,ohmask_l = Variable(img_l),Variable(mask_l),Variable(ohmask_l)
            else:
                img_l,mask_l,ohmask_l = Variable(img_l.cuda()),Variable(mask_l.cuda()),Variable(ohmask_l.cuda())

            #####################################
            #  labelled data Generator Training #
            #####################################
            # optimG.zero_grad()
            # optimD.zero_grad()
            cpmap = generator(img_l)
            N = cpmap.size()[0]
            H = cpmap.size()[2]
            W = cpmap.size()[3]
            # Generate the Real and Fake Labels
            targetf = Variable(torch.zeros((N,H,W))) # .long()
            targetr = Variable(torch.ones((N,H,W))) # .long()
            if not args.nogpu:
                targetf = targetf.cuda()
                targetr = targetr.cuda()

            cpmapsmax = nn.Softmax2d()(cpmap)
            cpmaplsmax = nn.LogSoftmax()(cpmap)
            conff = nn.Sigmoid()(discriminator(cpmapsmax))
            # LGce = nn.NLLLoss()(cpmaplsmax,mask_l)
            LGce = loss_calc(cpmaplsmax, mask_l)

            # LGadv = nn.BCELoss()(conff,mask_l)
            LGadv = loss_calc(cpmaplsmax, mask_l)
            LGadv_d = LGadv.data
            # LGadv_c = LGadv.cpu().numpy()
            LGce_d = LGce.data
            # LGce_c = LGce.cpu().numpy()

            # LGsemi_d = 0 # No semi-supervised training
            LGadv = args.lam_adv*LGadv
            (LGce + LGadv).backward()
            # optimG = poly_lr_scheduler(optimG, args.g_lr, itr)
            optimG.step()
            optimD.step()

            LGseg_d = LGce_d + LGadv_d + LGsemi_d
            # LGseg_c = LGce_c + LGadv_c + LGsemi_c

            # training_log = [epoch,itr,LD_c,LDr_c,LDf_c,LGseg_c,LGce_c,LGadv_c,LGsemi_c]
            print("[{}][{}] LGseg_d: {:.4f} LG_ce: {:.4f} LG_adv: {:.4f} LG_semi: {:.4f}"\
                    .format(epoch,itr,LGseg_d,LGce_d,LGadv_d,LGsemi_d))

        # best_miou = snapshot(generator,valoader,epoch,best_miou,args.snapshot_dir,args.prefix)
        best_miou, best_eiou = snapshot_segdis(generator,discriminator,valoader,epoch,best_miou,best_eiou,args.snapshot_dir,args.prefix)

'''
    Semi supervised training
'''
def train_joint(generator,discriminator,optimG,optimD,trainloader_l,valoader,args):

    best_miou = -1
    best_eiou = -1
    for epoch in range(args.start_epoch,args.max_epoch+1):
        generator.train()

        trainloader_l_iter = iter(trainloader_l)
        # trainloader_j_iter = iter(trainloader_j)
        print("Epoch: {}".format(epoch))
        batch_id = 0
        while(True):
            batch_id += 1
            itr = (len(trainloader_l))*(epoch-1) + batch_id
            optimG.zero_grad()
            optimD.zero_grad()
            optimD = poly_lr_scheduler(optimD, args.d_lr, itr)
            optimG = poly_lr_scheduler(optimG, args.g_lr, itr)

            LGjoint_d = 0
            loss_joint_value = 0
            loader_l = trainloader_l_iter
            try: 
                img_l,mask_l,ohmask_l,_ = next(loader_l)
            except:
                break

            if args.nogpu:
                img_l,mask_l,ohmask_l = Variable(img_l),Variable(mask_l),Variable(ohmask_l)
            else:
                img_l,mask_l,ohmask_l = Variable(img_l.cuda()),Variable(mask_l.cuda()),Variable(ohmask_l.cuda())

            if epoch > args.wait_semi:
                # semi unlabelled training
                cpmap = generator(img_l)
                cpmapsmax = nn.Softmax2d()(cpmap)
                # confsmax = nn.Softmax2d()(conf)
                conf = discriminator(cpmapsmax)
                confsigmoid = nn.Sigmoid()(conf)
                confsigmoid_remain = confsigmoid.detach().squeeze(1)
                # conflsmax = nn.LogSoftmax()(conf)

                # N = cpmap.size()[0]
                # H = cpmap.size()[2]
                # W = cpmap.size()[3]

                joint_gt = np.int8((confsigmoid_remain.cpu().numpy() >= args.t_semi))
                # print(joint_gt.shape)
                # semi_gt = cpmap.data.cpu().numpy().argmax(axis=1)
                # semi_gt[semi_ignore_mask] = 1
                joint_ratio = 1.0 - float(joint_gt.sum())/joint_gt.size
                # print('semi ratio: {:.4f}'.format(semi_ratio))
                if joint_ratio == 0.0:
                    loss_joint_value += 0
                else:
                    joint_gt = torch.FloatTensor(joint_gt)
                    LGjoint = args.lam_joint * loss_calc(cpmapsmax, joint_gt)
                    LGjoint_d += LGjoint.data.cpu().numpy()/args.lam_joint
                    # LGsemi += LGadv_d
                    LGjoint.backward()

            #####################################
            #  labelled data Generator Training #
            #####################################
            # optimG.zero_grad()
            # optimD.zero_grad()
            cpmap = generator(img_l)
            """
            N = cpmap.size()[0]
            H = cpmap.size()[2]
            W = cpmap.size()[3]
            # Generate the Real and Fake Labels
            targetf = Variable(torch.zeros((N,H,W))) # .long()
            targetr = Variable(torch.ones((N,H,W))) # .long()
            if not args.nogpu:
                targetf = targetf.cuda()
                targetr = targetr.cuda()
            """

            cpmapsmax = nn.Softmax2d()(cpmap)
            cpmaplsmax = nn.LogSoftmax()(cpmap)
            conff = nn.Sigmoid()(discriminator(cpmapsmax))
            # LGce = nn.NLLLoss()(cpmaplsmax,mask_l)
            LGce = loss_calc(cpmaplsmax, mask_l)

            # LGadv = nn.BCELoss()(conff,mask_l)
            LGadv = loss_calc(cpmaplsmax, mask_l)
            LGadv_d = LGadv.data
            # LGadv_c = LGadv.cpu().numpy()
            LGce_d = LGce.data
            # LGce_c = LGce.cpu().numpy()

            # LGsemi_d = 0 # No semi-supervised training
            LGadv = args.lam_adv*LGadv
            (LGce + LGadv).backward()
            # optimG = poly_lr_scheduler(optimG, args.g_lr, itr)
            optimG.step()
            optimD.step()

            LGseg_d = LGce_d + LGadv_d + LGjoint_d
            # LGseg_c = LGce_c + LGadv_c + LGsemi_c

            # training_log = [epoch,itr,LD_c,LDr_c,LDf_c,LGseg_c,LGce_c,LGadv_c,LGsemi_c]
            print("[{}][{}] LGseg_d: {:.4f} LG_ce: {:.4f} LG_adv: {:.4f} LG_joint: {:.4f}"\
                    .format(epoch,itr,LGseg_d,LGce_d,LGadv_d,LGjoint_d))

        # best_miou = snapshot(generator,valoader,epoch,best_miou,args.snapshot_dir,args.prefix)
        best_miou, best_eiou = snapshot_segdis(generator,discriminator,valoader,epoch,best_miou,best_eiou,args.snapshot_dir,args.prefix)


def main():
    args = parse_args()

    CUR_DIR = os.getcwd()
    with open(osp.join(CUR_DIR, "utils/config_crf.yaml")) as f:
        CRF_CONFIG = Dict(yaml.safe_load(f))

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
    # labtr = [IgnoreLabelClass(),ToTensorLabel(tensor_type=torch.FloatTensor)]
    # cotr = [RandomSizedCrop((320,320))] # (321,321)
    cotr = [RandomSizedCrop3((320,320))]

    print("dataset_dir: ", args.dataset_dir)
    if args.mode == 'semi':
        split_ratio = 0.8
    else:
        split_ratio = 1.0
    trainset_l = Corrosion(home_dir,args.dataset_dir,img_transform=Compose(imgtr), 
                           label_transform=Compose(labtr),co_transform=Compose(cotr),
                           split=split_ratio,labeled=True)
    trainloader_l = DataLoader(trainset_l,batch_size=args.batch_size,shuffle=True,
                               num_workers=2,drop_last=True)

    if args.mode == 'semi':
        trainset_u = Corrosion(home_dir,args.dataset_dir,img_transform=Compose(imgtr), 
                               label_transform=Compose(labtr),co_transform=Compose(cotr),
                               split=split_ratio,labeled=False)
        trainloader_u = DataLoader(trainset_u,batch_size=args.batch_size,shuffle=True,
                                   num_workers=2,drop_last=True)

    # if args.mode == 'joint':
        # trainloader_j = trainloader_l.copy()
        """
        trainset_j = Corrosion(home_dir,args.dataset_dir,img_transform=Compose(imgtr), 
                               label_transform=Compose(labtr),co_transform=Compose(cotr),
                               split=split_ratio,labeled=True)
        trainloader_j = DataLoader(trainset_j,batch_size=args.batch_size,shuffle=True,
                                   num_workers=2,drop_last=True)
        """

    postprocessor = DenseCRF(
        iter_max=CRF_CONFIG.CRF.ITER_MAX,
        pos_xy_std=CRF_CONFIG.CRF.POS_XY_STD,
        pos_w=CRF_CONFIG.CRF.POS_W,
        bi_xy_std=CRF_CONFIG.CRF.BI_XY_STD,
        bi_rgb_std=CRF_CONFIG.CRF.BI_RGB_STD,
        bi_w=CRF_CONFIG.CRF.BI_W,
    )

    #########################
    # Validation Dataloader #
    ########################
    if args.val_orig:
        if args.no_norm:
            imgtr = [ZeroPadding(),ToTensor()]
        else:
            imgtr = [ZeroPadding(),ToTensor(),NormalizeOwn()]
        labtr = [IgnoreLabelClass(),ToTensorLabel()]
        # labtr = [IgnoreLabelClass(),ToTensorLabel(tensor_type=torch.FloatTensor)]
        cotr = []
    else:
        if args.no_norm:
            imgtr = [ToTensor()]
        else:
            imgtr = [ToTensor(),NormalizeOwn()]
        labtr = [IgnoreLabelClass(),ToTensorLabel()]
        # labtr = [IgnoreLabelClass(),ToTensorLabel(tensor_type=torch.FloatTensor)]
        # cotr = [RandomSizedCrop3((320,320))] # (321,321)
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
    # model_summary = generator.cuda()

    init_weights(generator,args.init_net)

    if args.init_net != 'unet':
        optimG = optim.SGD(filter(lambda p: p.requires_grad, \
            generator.parameters()),lr=args.g_lr,momentum=0.9,\
            weight_decay=0.0001,nesterov=True)
    else:
        optimG = optim.Adam(filter(lambda p: p.requires_grad, \
            generator.parameters()),args.g_lr, [0.9, 0.999])

    if not args.nogpu:
        generator = nn.DataParallel(generator).cuda()

    #################
    # DISCRIMINATOR #
    ################
    if args.mode != "base":
        # softmax generator
        discriminator = DisSigmoid(in_channels=2)
        init_weights(generator,args.init_net)
        # model_summary = discriminator.cuda()
        # summary(model_summary, (2, 320, 320))
        if args.d_optim == 'adam':
            optimD = optim.Adam(filter(lambda p: p.requires_grad, \
                discriminator.parameters()),args.d_lr,[0.9,0.999])
                # discriminator.parameters()),[0.9,0.999],lr = args.d_lr,weight_decay=0.0001)
        else:
            optimD = optim.SGD(filter(lambda p: p.requires_grad, \
                discriminator.parameters()),lr=args.d_lr,weight_decay=0.0001,momentum=0.9,nesterov=True)

        if not args.nogpu:
            discriminator = nn.DataParallel(discriminator).cuda()

    if args.mode == 'base':
        train_base(generator,optimG,trainloader_l,valoader,args)
    elif args.mode == 'adv':
        train_adv(generator,discriminator,optimG,optimD,trainloader_l,valoader,postprocessor,args)
    elif args.mode == 'semi':
        train_semi(generator,discriminator,optimG,optimD,trainloader_l,trainloader_u,valoader,args)
    elif args.mode == 'joint':
        train_joint(generator,discriminator,optimG,optimD,trainloader_l,valoader,args)
    else:
        # train_semir(generator,discriminator,optimG,optimD,trainloader_l,valoader,args)
        print("training mode incorrect")

if __name__ == '__main__':

    main()
