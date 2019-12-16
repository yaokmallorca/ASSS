# from datasets.pascalvoc import PascalVOC
from datasets.corrosion import Corrosion
from torch.utils.data import DataLoader
import generators.deeplabv2 as deeplabv2
# import generators.fcn8s_soft as fcn
import torch
import torch.nn as nn
from torch.autograd import Variable
import argparse
import os
import numpy as np
from utils.metrics import scores
import torchvision.transforms as transforms
from utils.transforms import RandomSizedCrop, IgnoreLabelClass, ToTensorLabel, NormalizeOwn, ZeroPadding
from torchvision.transforms import ToTensor
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


def main():
    home_dir = os.path.dirname(os.path.realpath(__file__))

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir",help="A directory containing img (Images) \
                        and cls (GT Segmentation) folder")
    parser.add_argument("snapshot",help="Snapshot with the saved model")
    parser.add_argument("--val_orig", help="Do Inference on original size image.\
                        Otherwise, crop to 321x321 like in training ",action='store_true')
    parser.add_argument("--norm",help="Normalize the test images",\
                        action='store_true')
    args = parser.parse_args()
    # print(args.val_orig, args.norm)
    if args.val_orig:
        img_transform = transforms.Compose([ToTensor()])
        if args.norm:
            img_transform = transforms.Compose([ToTensor(),NormalizeOwn(dataset='corrosion')])
        label_transform = transforms.Compose([IgnoreLabelClass(),ToTensorLabel()])
        co_transform = transforms.Compose([RandomSizedCrop((321,321))])

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

    generator = deeplabv2.ResDeeplab()
    # generatro = fcn.FCN8s_soft()
    assert(os.path.isfile(args.snapshot))
    snapshot = torch.load(args.snapshot)

    saved_net = {k.partition('module.')[2]: v for i, (k,v) in enumerate(snapshot['state_dict'].items())}
    print('Snapshot Loaded')
    generator.load_state_dict(saved_net)
    generator.eval()
    generator = nn.DataParallel(generator).cuda()
    print('Generator Loaded')
    n_classes = 2

    gts, preds = [], []

    print('Prediction Goint to Start')
    colorize = VOCColorize()
    # TODO: Crop out the padding before prediction
    for img_id, (img, gt_mask, _, name) in enumerate(testloader):
        print("Generating Predictions for Image {}".format(img_id))
        gt_mask = gt_mask.numpy()[0]
        img = Variable(img.cuda())
        out_pred_map = generator(img)

        # Get hard prediction
        soft_pred = out_pred_map.data.cpu().numpy()[0]
        soft_pred = soft_pred[:,:gt_mask.shape[0],:gt_mask.shape[1]]
        hard_pred = np.argmax(soft_pred,axis=0).astype(np.uint8)

        print(hard_pred.shape, name)
        output = np.asarray(hard_pred, dtype=np.int)
        filename = os.path.join('results', '{}.png'.format(name[0]))
        color_file = Image.fromarray(colorize(output).transpose(1, 2, 0), 'RGB')
        color_file.save(filename)

        for gt_, pred_ in zip(gt_mask, hard_pred):
            gts.append(gt_)
            preds.append(pred_)
    score, class_iou = scores(gts, preds, n_class=n_classes)

    print("Mean IoU: {}".format(score))
if __name__ == '__main__':
    main()
