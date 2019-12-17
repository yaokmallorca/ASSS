import numpy as np
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose
home_dir = os.getcwd()
print(home_dir)
os.chdir(home_dir)

from utils.transforms import OneHotEncode

def load_image(file):
    return Image.open(file)

def read_img_list(filename):
    with open(filename) as f:
        img_list = []
        for line in f:
            img_list.append(line[:-1])
    return np.array(img_list)

class Corrosion(Dataset):

    TRAIN_LIST = "ImageSets/train.txt"
    VAL_LIST = "ImageSets/val.txt"

    def __init__(self, root, data_root, img_transform = Compose([]),\
     label_transform=Compose([]), co_transform=Compose([]),\
      train_phase=True,split=1,labeled=True,seed=0):
        np.random.seed(100)
        self.n_class = 2
        self.root = root
        self.data_root = data_root
        self.images_root = os.path.join(self.data_root, 'corrosion', 'JPEGImages')
        # print("images_root: ", self.images_root)
        self.labels_root = os.path.join(self.data_root, 'corrosion', 'SemanticLabels')
        # print("labels_root: ", self.labels_root)
        self.elabels_root = os.path.join(self.data_root, 'corrosion', 'EvaluateLabels')
        self.img_list = read_img_list(os.path.join(self.data_root,'corrosion',self.TRAIN_LIST)) \
                        if train_phase else read_img_list(os.path.join(self.data_root,'corrosion',self.VAL_LIST))
        self.split = split
        self.labeled = labeled
        n_images = len(self.img_list)
        self.img_l = np.random.choice(range(n_images),int(n_images*split),replace=False) # Labeled Images
        self.img_u = np.array([idx for idx in range(n_images) if idx not in self.img_l],dtype=int) # Unlabeled Images
        if self.labeled:
            self.img_list = self.img_list[self.img_l]
        else:
            self.img_list = self.img_list[self.img_u]
        self.img_transform = img_transform
        self.label_transform = label_transform
        self.co_transform = co_transform
        self.train_phase = train_phase

    def __getitem__(self, index):
        filename = self.img_list[index]

        with open(os.path.join(self.images_root,filename+'.jpg'), 'rb') as f:
            # print(f)
            image = load_image(f).convert('RGB')
        with open(os.path.join(self.labels_root,filename+'.bmp'), 'rb') as f:
            # print(f)
            label = load_image(f).convert('L')
        with open(os.path.join(self.elabels_root,filename+'.bmp'), 'rb') as f:
            # print(f)
            elabel = load_image(f).convert('L')

        # image, label = self.co_transform((image, label))
        image, label, elabel = self.co_transform((image, label, elabel))
        image = self.img_transform(image)
        label = self.label_transform(label)
        elabel = self.label_transform(elabel)
        ohlabel = OneHotEncode()(label)

        if self.train_phase:
            return image, label, ohlabel, elabel
        else:
            return image, label, ohlabel, elabel, filename

    def __len__(self):
        return len(self.img_list)


def test():
    from utils.transforms import RandomSizedCrop, IgnoreLabelClass, ToTensorLabel, NormalizeOwn,ZeroPadding, OneHotEncode, RandomSizedCrop3
    from torchvision.transforms import ToTensor,Compose
    import matplotlib.pyplot as plt

    imgtr = [ToTensor(),NormalizeOwn()]
    # sigmoid 
    labtr = [IgnoreLabelClass(),ToTensorLabel(tensor_type=torch.FloatTensor)]
    # cotr = [RandomSizedCrop((320,320))] # (321,321)
    cotr = [RandomSizedCrop3((320,320))]

    dataset_dir = '/media/data/seg_dataset'
    trainset = Corrosion(home_dir, dataset_dir,img_transform=Compose(imgtr), 
                           label_transform=Compose(labtr),co_transform=Compose(cotr),
                           split=args.split,labeled=True)
    trainloader = DataLoader(trainset_l,batch_size=1,shuffle=True,
                               num_workers=2,drop_last=True)

    for batch_id, (img, mask, _, emask) in enumerate(trainloader):
        img, mask, emask = img.numpy()[0], mask.numpy()[0], emask.numpy()[0]
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        ax1.imshow(img)
        ax2.imshow(mask)
        ax3.imshow(emask)
        plt.show()

if __name__ == '__main__':
    test()