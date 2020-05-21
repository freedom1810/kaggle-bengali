from albumentations import Compose, Resize, Rotate, Normalize, OneOf, ElasticTransform, GridDistortion, OpticalDistortion, CoarseDropout
import numpy as np
from skimage.transform import AffineTransform, warp
import albumentations as A
import cv2
from .auto_aug import *
from PIL import Image, ImageEnhance, ImageOps

def train_aug(image_size, use_cutout = False):
    if use_cutout:
        return Compose([Resize(*image_size),
                        Rotate(10),
                        # HorizontalFlip(),
                        OneOf([
                            ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
                            GridDistortion()

                        ], p=0.5),
                        CoarseDropout(max_holes=10, min_holes = 3,max_height=12, max_width=12),
                        Normalize()], p = 1)

    return Compose([Resize(*image_size),
                    Rotate(10),
                    # HorizontalFlip(),
                    OneOf([
                        ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
                        GridDistortion()
                    ], p=0.5),
                    Normalize()], p = 1)
    


def valid_aug(image_size):
    
    return Compose([Resize(*image_size),
                    Normalize()], p = 1)


# --- LAFOSS is God
def bbox(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, rmax, cmin, cmax


def crop_resize(img0, size=(128,128), pad=16):
    HEIGHT = 137
    WIDTH = 236
    SIZE = 128
    #crop a box around pixels large than the threshold 
    #some images contain line at the sides
    ymin,ymax,xmin,xmax = bbox(img0[5:-5,5:-5] > 80)
    #cropping may cut too much, so we need to add it back
    xmin = xmin - 13 if (xmin > 13) else 0
    ymin = ymin - 10 if (ymin > 10) else 0
    xmax = xmax + 13 if (xmax < WIDTH - 13) else WIDTH
    ymax = ymax + 10 if (ymax < HEIGHT - 10) else HEIGHT
    img = img0[ymin:ymax,xmin:xmax]
    #remove lo intensity pixels as noise
    # img[img < 28] = 0 bỏ dòng này để dùng auto_aug
    lx, ly = xmax-xmin,ymax-ymin
    l = max(lx,ly) + pad
    #make sure that the aspect ratio is kept in rescaling
    img = np.pad(img, [((l-ly)//2,), ((l-lx)//2,)], mode='constant')
    return cv2.resize(img,size)
    # return img

# --- LAFOSS is God

def apply_aug(aug, image):
    return aug(image=image)['image']    



class Transform:
    def __init__(self,
                path = '',
                train=False,
                mode = 'train',
                crop=False,
                padding = 16,
                image_size=(224, 224), 
                use_cutout = False, 
                auto_aug = False
                ):

        self.path = path
        self.train = train
        self.mode = mode
        self.crop = crop
        self.padding = padding
        self.auto_aug = auto_aug

        self.image_size = image_size
        self.use_cutout = use_cutout

        if self.auto_aug:
            self.trans = AutoAugment()

    def __call__(self, example):
        if self.train:
            path, y = example
        else:
            path = example

        if self.crop:
            img = cv2.imread(self.path + path + '.jpg', 0)
            img = (img*(255.0/img.max())).astype(np.uint8)
            img = crop_resize(img, size= self.image_size, pad = self.padding)
            img = cv2.merge((img, img, img))
        else:
            img = cv2.imread(self.path + path + '.png', 1)


        if self.auto_aug:
            # auto_aug
            img = Image.fromarray(img).convert("RGB")
            img = self.trans(img)
            img = np.array(img)


        if self.mode == 'train':
            img = apply_aug(train_aug(self.image_size, self.use_cutout), img)
        else:
            img = apply_aug(valid_aug(self.image_size), img)

        # cv2.imwrite('test_auto_aug/' + path + '.jpg', img)
        img = np.moveaxis(img, -1, 0)  # conver to channel first, pytorch suck
        img = img.astype(np.float32)

        if self.train:
            y = y.astype(np.int64)
            return img, y
        else:
            return img


