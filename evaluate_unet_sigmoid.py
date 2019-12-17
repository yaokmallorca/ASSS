# from datasets.pascalvoc import PascalVOC
from datasets.corrosion import Corrosion
from torch.utils.data import DataLoader
import generators.unet as unet
from discriminators.discriminator_unet import Dis
# import generators.deeplabv2 as deeplabv2
# import generators.fcn8s_soft as fcn
import torch
import torch.nn as nn
from torch.autograd import Variable
import argparse
import os
import os.path as osp
import numpy as np
from utils.metrics import scores
import torchvision.transforms as transforms
from utils.transforms import ResizedImage, IgnoreLabelClass, ToTensorLabel, NormalizeOwn, ZeroPadding, ResizedImage3
from torchvision.transforms import ToTensor
import cv2
from PIL import Image

class VOCColorize(object):
    def __init__(self, n=22):
        self.cmap = color_map(22)
        self.cmap = torch.from_numpy(self.cmap[:n])

    def __call__(self, gray_image):
        size = gray_image.shape
        color_image = np.zeros((3, size[0], size[1]), dtype=np.uint8)
        for label in range(0, len(self.cmap)):
            mask = (label == gray_image)
            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]

        # handle void
        mask = (0 == gray_image) # 255
        color_image[0][mask] = color_image[1][mask] = color_image[2][mask] = 0 # 255

        return color_image

def color_map(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap

def show_all(gt, pred):
    import matplotlib.pyplot as plt
    from matplotlib import colors
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    fig, axes = plt.subplots(1, 2)
    ax1, ax2 = axes

    classes = np.array(('background',  'corrosion'))
    colormap = [(0,0,0), (0.5,0,0)]
    cmap = colors.ListedColormap(colormap)
    # bounds=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]
    bounds=[0,1]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    ax1.set_title('gt')
    ax1.imshow(gt, cmap=cmap, norm=norm)

    ax2.set_title('pred')
    ax2.imshow(pred, cmap=cmap, norm=norm)

    plt.show()

def make_palette(num_classes):
    """
    Maps classes to colors in the style of PASCAL VOC.
    Close values are mapped to far colors for segmentation visualization.
    See http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html#devkit

    Takes:
        num_classes: the number of classes
    Gives:
        palette: the colormap as a k x 3 array of RGB colors
    """
    palette = np.zeros((num_classes, 3), dtype=np.uint8)
    for k in range(0, num_classes):
        label = k
        i = 0
        while label:
            palette[k, 0] |= (((label >> 0) & 1) << (7 - i))
            palette[k, 1] |= (((label >> 1) & 1) << (7 - i))
            palette[k, 2] |= (((label >> 2) & 1) << (7 - i))
            label >>= 3
            i += 1
    return palette

def vis_seg(img, seg, palette, alpha=0.5):
    """
    Visualize segmentation as an overlay on the image.

    Takes:
        img: H x W x 3 image in [0, 255]
        seg: H x W segmentation image of class IDs
        palette: K x 3 colormap for all classes
        alpha: opacity of the segmentation in [0, 1]
    Gives:
        H x W x 3 image with overlaid segmentation
    """
    vis = np.array(img, dtype=np.float32)
    mask = seg > 0
    vis[mask] *= 1. - alpha
    vis[mask] += alpha * palette[seg[mask].flat]
    vis = vis.astype(np.uint8)
    return vis

def evaluate_generator():
    home_dir = os.path.dirname(os.path.realpath(__file__))

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir",help="A directory containing img (Images) \
                        and cls (GT Segmentation) folder")
    parser.add_argument("snapshot",help="Snapshot with the saved model")
    parser.add_argument("--val_orig", help="Do Inference on original size image.\
                        Otherwise, crop to 320x320 like in training ",action='store_true')
    parser.add_argument("--norm",help="Normalize the test images",\
                        action='store_true')
    args = parser.parse_args()
    # print(args.val_orig, args.norm)
    if args.val_orig:
        img_transform = transforms.Compose([ToTensor()])
        if args.norm:
            img_transform = transforms.Compose([ToTensor(),NormalizeOwn(dataset='corrosion')])
        label_transform = transforms.Compose([IgnoreLabelClass(),ToTensorLabel()])
        # co_transform = transforms.Compose([RandomSizedCrop((320,320))])
        co_transform = transforms.Compose([ResizedImage3((320,320))])

        testset = Corrosion(home_dir, args.dataset_dir,img_transform=img_transform, \
            label_transform = label_transform,co_transform=co_transform,train_phase=False)
        testloader = DataLoader(testset, batch_size=1)
    else:
        img_transform = transforms.Compose([ZeroPadding(),ToTensor()])
        if args.norm:
            img_transform = img_transform = transforms.Compose([ZeroPadding(),ToTensor(),NormalizeOwn(dataset='corrosion')])
        label_transform = transforms.Compose([IgnoreLabelClass(),ToTensorLabel()])

        testset = Corrosion(home_dir,args.dataset_dir,img_transform=img_transform, \
            label_transform=label_transform,train_phase=False)
        testloader = DataLoader(testset, batch_size=1)

    # generator = deeplabv2.ResDeeplab()
    # generatro = fcn.FCN8s_soft()
    generator = unet.AttU_Net(output_ch=1)
    print(args.snapshot)
    assert(os.path.isfile(args.snapshot))
    snapshot = torch.load(args.snapshot)

    saved_net = {k.partition('module.')[2]: v for i, (k,v) in enumerate(snapshot['state_dict'].items())}
    print('Snapshot Loaded')
    generator.load_state_dict(saved_net)
    generator.eval()
    generator = nn.DataParallel(generator).cuda()
    print('Generator Loaded')
    n_classes = 2

    gts, preds, gtes = [], [], []

    print('Prediction Goint to Start')
    colorize = VOCColorize()
    palette = make_palette(2)
    # print(palette)
    IMG_DIR = osp.join(args.dataset_dir, 'corrosion/JPEGImages')
    # TODO: Crop out the padding before prediction
    for img_id, (img, gt_mask, _, gte_mask, name) in enumerate(testloader):
        print("Generating Predictions for Image {}".format(img_id))
        gt_mask = gt_mask.numpy()[0]
        gte_mask = gte_mask.numpy()[0]
        img = Variable(img.cuda())
        # img.cpu().numpy()[0]
        img_path = osp.join(IMG_DIR, name[0]+'.jpg')
        print(img_path)
        img_array = cv2.imread(img_path)
        img_array = cv2.resize(img_array, (320,320), interpolation = cv2.INTER_AREA) 
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        out_pred_map = generator(img)

        # Get hard prediction
        soft_pred = out_pred_map.data.cpu().numpy()[0]
        soft_pred = soft_pred[:,:gt_mask.shape[0],:gt_mask.shape[1]]
        # hard_pred = np.argmax(soft_pred,axis=0).astype(np.uint8)
        soft_pred[soft_pred>=0.5] = 1
        soft_pred[soft_pred<0.5] = 0
        hard_pred = soft_pred

        output = np.asarray(hard_pred, dtype=np.int)
        print(output.shape, name)
        filename = os.path.join('results', '{}.png'.format(name[0]))
        color_file = Image.fromarray(colorize(output[0]).transpose(1, 2, 0), 'RGB')
        color_file.save(filename)

        masked_im = Image.fromarray(vis_seg(img_array, output[0], palette))
        masked_im.save(filename[0:-4] + '_vis.jpg')

        for gt_, gte_, pred_ in zip(gt_mask, gte_mask, output):
            gts.append(gt_)
            preds.append(pred_[0])
            gtes.append(gte_)
    score, class_iou = scores(gts, preds, n_class=n_classes)
    escore, _ = scores(gtes, preds, n_class=n_classes)
    print("Mean IoU: {}, Mean eIou: {}".format(score, escore))


def evaluate_discriminator():
    home_dir = os.path.dirname(os.path.realpath(__file__))

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir",help="A directory containing img (Images) \
                        and cls (GT Segmentation) folder")
    parser.add_argument("snapshot_g",help="Snapshot with the saved generator model")
    parser.add_argument("snapshot_d",help="Snapshot with the saved discriminator model")
    parser.add_argument("--val_orig", help="Do Inference on original size image.\
                        Otherwise, crop to 320x320 like in training ",action='store_true')
    parser.add_argument("--norm",help="Normalize the test images",\
                        action='store_true')
    args = parser.parse_args()
    # print(args.val_orig, args.norm)
    if args.val_orig:
        img_transform = transforms.Compose([ToTensor()])
        if args.norm:
            img_transform = transforms.Compose([ToTensor(),NormalizeOwn(dataset='corrosion')])
        label_transform = transforms.Compose([IgnoreLabelClass(),ToTensorLabel()])
        # co_transform = transforms.Compose([RandomSizedCrop((320,320))])
        co_transform = transforms.Compose([ResizedImage((320,320))])

        testset = Corrosion(home_dir, args.dataset_dir,img_transform=img_transform, \
            label_transform = label_transform,co_transform=co_transform,train_phase=False)
        testloader = DataLoader(testset, batch_size=1)
    else:
        img_transform = transforms.Compose([ZeroPadding(),ToTensor()])
        if args.norm:
            img_transform = img_transform = transforms.Compose([ZeroPadding(),ToTensor(),NormalizeOwn(dataset='corrosion')])
        label_transform = transforms.Compose([IgnoreLabelClass(),ToTensorLabel()])

        testset = Corrosion(home_dir,args.dataset_dir,img_transform=img_transform, \
            label_transform=label_transform,train_phase=False)
        testloader = DataLoader(testset, batch_size=1)

    # generator = deeplabv2.ResDeeplab()
    # generatro = fcn.FCN8s_soft()
    generator = unet.AttU_Net()
    print(args.snapshot_g)
    assert(os.path.isfile(args.snapshot_g))
    snapshot_g = torch.load(args.snapshot_g)

    discriminator = Dis(in_channels=2)
    print(args.snapshot_d)
    assert(os.path.isfile(args.snapshot_d))
    snapshot_d = torch.load(args.snapshot_d)

    saved_net = {k.partition('module.')[2]: v for i, (k,v) in enumerate(snapshot_g['state_dict'].items())}
    print('Generator Snapshot Loaded')
    generator.load_state_dict(saved_net)
    generator.eval()
    generator = nn.DataParallel(generator).cuda()
    print('Generator Loaded')

    saved_net_d = {k.partition('module.')[2]: v for i, (k,v) in enumerate(snapshot_d['state_dict'].items())}
    print('Discriminator Snapshot Loaded')
    discriminator.load_state_dict(saved_net_d)
    discriminator.eval()
    discriminator = nn.DataParallel(discriminator).cuda()
    print('discriminator Loaded')
    n_classes = 2

    gts, preds = [], []
    print('Prediction Goint to Start')
    colorize = VOCColorize()
    palette = make_palette(2)
    # print(palette)
    IMG_DIR = osp.join(args.dataset_dir, 'corrosion/JPEGImages')
    # TODO: Crop out the padding before prediction
    for img_id, (img, gt_mask, _, gte_mask, name) in enumerate(testloader):
        print("Generating Predictions for Image {}".format(img_id))
        gt_mask = gt_mask.numpy()[0]
        img = Variable(img.cuda())
        # img.cpu().numpy()[0]
        img_path = osp.join(IMG_DIR, name[0]+'.jpg')
        print(img_path)
        img_array = cv2.imread(img_path)
        img_array = cv2.resize(img_array, (320,320), interpolation = cv2.INTER_AREA) 
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        out_pred_map = generator(img)
        # print(out_pred_map.size())

        # Get hard prediction
        soft_pred = out_pred_map.data.cpu().numpy()[0]
        # print("gen: ", soft_pred.shape)
        # print(soft_pred.shape)
        soft_pred = soft_pred[:,:gt_mask.shape[0],:gt_mask.shape[1]]
        # print("gen: ", soft_pred.shape)
        # print(soft_pred.shape)
        hard_pred = np.argmax(soft_pred,axis=0).astype(np.uint8)
        # print("gen: ", hard_pred.shape)

        # Get discriminator prediction
        dis_conf = discriminator(out_pred_map)
        dis_confsmax = nn.Softmax2d()(dis_conf)
        # print(dis_conf.size())
        dis_soft_pred = dis_confsmax.data.cpu().numpy()[0]
        # dis_soft_pred[dis_soft_pred<=0.2] = 0
        # dis_soft_pred[dis_soft_pred>0.2] = 1
        # print("dis: ", dis_soft_pred.shape)
        dis_hard_pred = np.argmax(dis_soft_pred,axis=0).astype(np.uint8)
        # print("dis: ", dis_hard_pred.shape)
        # dis_pred = dis_pred[:,:gt_mask.shape[0],:gt_mask.shape[1]]
        # print(soft_pred.shape)
        # dis_hard_pred = np.argmax(dis_pred,axis=0).astype(np.uint8)

        # print(hard_pred.shape, name)
        output = np.asarray(hard_pred, dtype=np.int)
        # print("gen: ", output.shape)
        filename = os.path.join('results', '{}.png'.format(name[0]))
        color_file = Image.fromarray(colorize(output).transpose(1, 2, 0), 'RGB')
        color_file.save(filename)

        masked_im = Image.fromarray(vis_seg(img_array, output, palette))
        masked_im.save(filename[0:-4] + '_vis.png')

        # discriminator output
        dis_output = np.asarray(dis_hard_pred, dtype=np.int)
        # print("dis: ", dis_output.shape)
        dis_filename = os.path.join('results', '{}_dis.png'.format(name[0][0:-4]))
        dis_color_file = Image.fromarray(colorize(dis_output).transpose(1, 2, 0), 'RGB')
        dis_color_file.save(dis_filename)

        for gt_, pred_ in zip(gt_mask, hard_pred):
            gts.append(gt_)
            preds.append(pred_)
        # input('s')
    score, class_iou = scores(gts, preds, n_class=n_classes)
    print("Mean IoU: {}".format(score))


if __name__ == '__main__':
    # main()
    # evaluate_discriminator()
    evaluate_generator()
