# Copyright (c) 2015-present, Facebook, Inc. Modified By Wenhao Wang, Weipu Zhang.
# All rights reserved.
import warnings
warnings.filterwarnings('ignore')
from models.gem import GeneralizedMeanPoolingP
import sys
import os
import pdb
import argparse
import time
from collections import OrderedDict, defaultdict

from PIL import Image
from isc.io import write_hdf5_descriptors

import torch
import torchvision
import torchvision.transforms
from torch import nn
from torch.utils.data import Dataset

import faiss
import tempfile
import numpy as np
import h5py

class GaussianBlur(object):
    def __init__(self, sigma=[5.0, 5.0]):
        self.sigma = sigma

    def __call__(self, x):
        import random
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

def load_model(name, checkpoint_file, GeM_p):
    
    if name == "50":
        import models
        model = models.create('resnet50', num_features=0, dropout=0, num_classes=4)
        model = nn.DataParallel(model)
        model.module.classifier = None
        model.module.classifier_1 = None
        #import pickle as pkl
        print("loading ckpt... ", checkpoint_file)
        #para = pkl.load(open('./' +  checkpoint_file, 'rb'), encoding='utf-8')['model']
        ckpt = torch.load(checkpoint_file)
        mkg = model.load_state_dict(ckpt,strict = False)
        print(mkg)
        #model.module.base[10] =  GeneralizedMeanPoolingP(3)
        model.eval()
        return model
    
    if name == "50S":
        import models
        model = models.create('resneSt50', num_features=0, dropout=0, num_classes=4)
        model = nn.DataParallel(model)
        model.module.classifier = None
        model.module.classifier_1 = None
        #import pickle as pkl
        print("loading ckpt... ", checkpoint_file)
        #para = pkl.load(open('./' +  checkpoint_file, 'rb'), encoding='utf-8')['model']
        ckpt = torch.load(checkpoint_file)
        model.load_state_dict(ckpt,strict = True)
        #model.module.base[10] =  GeneralizedMeanPoolingP(3)
        model.eval()
        return model
    
    if name == "50HS":
        import models
        model = models.create('hsresnet50', num_features=0, dropout=0, num_classes=4)
        model = nn.DataParallel(model)
        model.module.classifier = None
        model.module.classifier_1 = None
        #import pickle as pkl
        print("loading ckpt... ", checkpoint_file)
        #para = pkl.load(open('./' +  checkpoint_file, 'rb'), encoding='utf-8')['model']
        ckpt = torch.load(checkpoint_file)
        model.load_state_dict(ckpt,strict = True)
        #model.module.base[10] =  GeneralizedMeanPoolingP(3)
        model.eval()
        return model
    
    if name == "50IBN":
        import models
        model = models.create('resnet_ibn50a', num_features=0, dropout=0, num_classes=4)
        model = nn.DataParallel(model)
        model.module.classifier = None
        model.module.classifier_1 = None
        #import pickle as pkl
        print("loading ckpt... ", checkpoint_file)
        #para = pkl.load(open('./' +  checkpoint_file, 'rb'), encoding='utf-8')['model']
        ckpt = torch.load(checkpoint_file)
        model.load_state_dict(ckpt,strict = True)
        #model.module.base[10] =  GeneralizedMeanPoolingP(3)
        model.eval()
        return model
    
    if name == "50X":
        import models
        model = models.create('resneXt50', num_features=0, dropout=0, num_classes=4)
        model = nn.DataParallel(model)
        model.module.classifier = None
        model.module.classifier_1 = None
        #import pickle as pkl
        print("loading ckpt... ", checkpoint_file)
        #para = pkl.load(open('./' +  checkpoint_file, 'rb'), encoding='utf-8')['model']
        ckpt = torch.load(checkpoint_file)
        mkg = model.load_state_dict(ckpt,strict = False)
        print(mkg)
        #model.module.base[10] =  GeneralizedMeanPoolingP(3)
        model.eval()
        return model
    
    if name == "50C":
        import models
        model = models.create('cotnet50', num_features=0, dropout=0, num_classes=4)
        model = nn.DataParallel(model)
        model.module.classifier = None
        model.module.classifier_1 = None
        #import pickle as pkl
        print("loading ckpt... ", checkpoint_file)
        #para = pkl.load(open('./' +  checkpoint_file, 'rb'), encoding='utf-8')['model']
        ckpt = torch.load(checkpoint_file)
        model.load_state_dict(ckpt,strict = True)
        #model.module.base[10] =  GeneralizedMeanPoolingP(3)
        model.eval()
        return model
    
    if name == "CAE":
        import models
        model = models.create('cae_base', num_features=0, dropout=0, num_classes=4)
        model = nn.DataParallel(model)
        model.module.classifier = None
        model.module.classifier_1 = None
        #import pickle as pkl
        print("loading ckpt... ", checkpoint_file)
        #para = pkl.load(open('./' +  checkpoint_file, 'rb'), encoding='utf-8')['model']
        ckpt = torch.load(checkpoint_file)
        model.load_state_dict(ckpt,strict = True)
        #model.module.base[10] =  GeneralizedMeanPoolingP(3)
        model.eval()
        return model
    
    if name == "50P":
        import models
        model = models.create('pyresnet50', num_features=0, dropout=0, num_classes=4)
        model = nn.DataParallel(model)
        model.module.classifier = None
        model.module.classifier_1 = None
        #import pickle as pkl
        print("loading ckpt... ", checkpoint_file)
        #para = pkl.load(open('./' +  checkpoint_file, 'rb'), encoding='utf-8')['model']
        ckpt = torch.load(checkpoint_file)
        model.load_state_dict(ckpt,strict = True)
        #model.module.base[10] =  GeneralizedMeanPoolingP(3)
        model.eval()
        return model
    
    if name == "50CB":
        import models
        model = models.create('resnet50_cbam', num_features=0, dropout=0, num_classes=4)
        model = nn.DataParallel(model)
        model.module.classifier = None
        model.module.classifier_1 = None
        #import pickle as pkl
        print("loading ckpt... ", checkpoint_file)
        #para = pkl.load(open('./' +  checkpoint_file, 'rb'), encoding='utf-8')['model']
        ckpt = torch.load(checkpoint_file)
        model.load_state_dict(ckpt,strict = True)
        #model.module.base[10] =  GeneralizedMeanPoolingP(3)
        model.eval()
        return model
    
    if name == "50I":
        import models
        model = models.create('iresnet50', num_features=0, dropout=0, num_classes=4)
        model = nn.DataParallel(model)
        model.module.classifier = None
        model.module.classifier_1 = None
        #import pickle as pkl
        print("loading ckpt... ", checkpoint_file)
        #para = pkl.load(open('./' +  checkpoint_file, 'rb'), encoding='utf-8')['model']
        ckpt = torch.load(checkpoint_file)
        model.load_state_dict(ckpt,strict = True)
        #model.module.base[10] =  GeneralizedMeanPoolingP(3)
        model.eval()
        return model
    
    if name == "50SK":
        import models
        model = models.create('sknet50', num_features=0, dropout=0, num_classes=4)
        model = nn.DataParallel(model)
        model.module.classifier = None
        model.module.classifier_1 = None
        #import pickle as pkl
        print("loading ckpt... ", checkpoint_file)
        #para = pkl.load(open('./' +  checkpoint_file, 'rb'), encoding='utf-8')['model']
        ckpt = torch.load(checkpoint_file)
        mkg = model.load_state_dict(ckpt,strict = False)
        print(mkg)
        #model.module.base[10] =  GeneralizedMeanPoolingP(3)
        model.eval()
        return model
    
    if name == "vit_base":
        import models
        model = models.create('vit_base', num_features=0, dropout=0, num_classes=4)
        model = nn.DataParallel(model)
        model.module.classifier = None
        model.module.classifier_1 = None
        #import pickle as pkl
        print("loading ckpt... ", checkpoint_file)
        #para = pkl.load(open('./' +  checkpoint_file, 'rb'), encoding='utf-8')['model']
        ckpt = torch.load(checkpoint_file)
        mkg = model.load_state_dict(ckpt,strict = False)
        print(mkg)
        #model.module.base[10] =  GeneralizedMeanPoolingP(3)
        model.eval()
        return model
    
    if name == "t2t":
        import models
        model = models.create('T2T_vit_14', num_features=0, dropout=0, num_classes=4)
        model = nn.DataParallel(model)
        model.module.classifier = None
        model.module.classifier_1 = None
        #import pickle as pkl
        print("loading ckpt... ", checkpoint_file)
        #para = pkl.load(open('./' +  checkpoint_file, 'rb'), encoding='utf-8')['model']
        ckpt = torch.load(checkpoint_file)
        mkg = model.load_state_dict(ckpt,strict = False)
        print(mkg)
        #model.module.base[10] =  GeneralizedMeanPoolingP(3)
        model.eval()
        return model
    
    if name == "swin_base":
        import models
        model = models.create('swin_base', num_features=0, dropout=0, num_classes=4)
        model = nn.DataParallel(model)
        model.module.classifier = None
        model.module.classifier_1 = None
        #import pickle as pkl
        print("loading ckpt... ", checkpoint_file)
        #para = pkl.load(open('./' +  checkpoint_file, 'rb'), encoding='utf-8')['model']
        ckpt = torch.load(checkpoint_file)
        mkg = model.load_state_dict(ckpt,strict = False)
        print(mkg)
        #model.module.base[10] =  GeneralizedMeanPoolingP(3)
        model.eval()
        return model
    
    if name == "PVT_v2_b2":
        import models
        model = models.create('PVT_v2_b2', num_features=0, dropout=0, num_classes=4)
        model = nn.DataParallel(model)
        model.module.classifier = None
        model.module.classifier_1 = None
        #import pickle as pkl
        print("loading ckpt... ", checkpoint_file)
        #para = pkl.load(open('./' +  checkpoint_file, 'rb'), encoding='utf-8')['model']
        ckpt = torch.load(checkpoint_file)
        model.load_state_dict(ckpt,strict = True)
        #model.module.base[10] =  GeneralizedMeanPoolingP(3)
        model.eval()
        return model
    
    if name == "uni_base":
        import models
        model = models.create('uni_base', num_features=0, dropout=0, num_classes=4)
        model = nn.DataParallel(model)
        model.module.classifier = None
        model.module.classifier_1 = None
        #import pickle as pkl
        print("loading ckpt... ", checkpoint_file)
        #para = pkl.load(open('./' +  checkpoint_file, 'rb'), encoding='utf-8')['model']
        ckpt = torch.load(checkpoint_file)
        model.load_state_dict(ckpt,strict = True)
        #model.module.base[10] =  GeneralizedMeanPoolingP(3)
        model.eval()
        return model
    
    if name == "NAT_base":
        import models
        model = models.create('NAT_base', num_features=0, dropout=0, num_classes=4)
        model = nn.DataParallel(model)
        model.module.classifier = None
        model.module.classifier_1 = None
        #import pickle as pkl
        print("loading ckpt... ", checkpoint_file)
        #para = pkl.load(open('./' +  checkpoint_file, 'rb'), encoding='utf-8')['model']
        ckpt = torch.load(checkpoint_file)
        model.load_state_dict(ckpt,strict = True)
        #model.module.base[10] =  GeneralizedMeanPoolingP(3)
        model.eval()
        return model

    assert False

class ImageList(Dataset):

    def __init__(self, image_list, blur, bw, imsize=None, transform=None):
        Dataset.__init__(self)
        self.image_list = image_list
        self.transform = transform
        self.imsize = imsize
        self.blur = blur
        self.bw = bw

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, i):
        x = Image.open(self.image_list[i])
        x = x.convert("RGB")
        if self.blur:
            x = GaussianBlur([5, 5])(x)
        if self.bw:
            #print("black-white")
            x = torchvision.transforms.RandomGrayscale(p=1)(x)
            #x = x.convert("L")
        if self.imsize is not None:
            x = x.resize((self.imsize,self.imsize))
        if self.transform is not None:
            x = self.transform(x)
        return x


def main():

    parser = argparse.ArgumentParser()

    def aa(*args, **kwargs):
        group.add_argument(*args, **kwargs)

    group = parser.add_argument_group('feature extraction options')
    aa('--transpose', default=-1, type=int, help="one of the 7 PIL transpose options ")
    aa('--train_pca', default=False, action="store_true", help="run PCA training")
    aa('--pca_file', default="", help="File with PCA descriptors")
    aa('--pca_dim', default=1500, type=int, help="output dimension for PCA")
    aa('--device', default="cpu", help='pytroch device')
    aa('--batch_size', default=500, type=int, help="max batch size to use for extraction")
    aa('--num_workers', default=20, type=int, help="nb of dataloader workers")

    group = parser.add_argument_group('model options')
    aa('--model', default='multigrain_resnet50', help="model to use")
    aa('--checkpoint', default='data/multigrain_joint_3B_0.5.pth', help='override default checkpoint')
    aa('--GeM_p', default=7.0, type=float, help="Power used for GeM pooling")
    aa('--scales', default="1.0", help="scale levels")
    aa('--imsize', default=512, type=int, help="max image size at extraction time")

    group = parser.add_argument_group('dataset options')
    aa('--image_dir', default="", help="search image files in these directories")
    aa('--n_train_pca', default=10000, type=int, help="nb of training vectors for the PCA")
    aa('--blur', default=False, action="store_true", help="whether using blur augmentation")
    aa('--bw', default=False, action="store_true", help="whether using black-white augmentation")

    group = parser.add_argument_group('output options')
    aa('--o', default="/tmp/desc.hdf5", help="write trained features to this file")

    args = parser.parse_args()
    args.scales = [float(x) for x in args.scales.split(",")]

    print("args=", args)
    #print("reading image names from", args.file_list)
    #image_list = [l.strip() for l in open(args.file_list, "r")]

    # add jpg suffix if there is none
    #image_list = [
    #    fname if "." in fname else fname + ".jpg"
    #    for fname in image_list
    #]

    # full path name for the image
    image_dir = args.image_dir
    if not image_dir.endswith('/'):
        image_dir += "/"

    #image_list = [image_dir + fname for fname in image_list]
    namess = os.listdir(image_dir)
    image_list = sorted([image_dir + '/' + name for name in namess])

    print(f"  found {len(image_list)} images")

    if args.train_pca:
        rs = np.random.RandomState(123)
        image_list = [
            image_list[i]
            for i in rs.choice(len(image_list), size=args.n_train_pca, replace=False)
        ]
        print(f"subsampled {args.n_train_pca} vectors")

    # transform without resizing
    mean, std = [0.485, 0.456, 0.406],[0.229, 0.224, 0.225]

    transforms = [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean, std)
    ]

    if args.transpose != -1:
        transforms.insert(TransposeTransform(args.transpose), 0)

    transforms = torchvision.transforms.Compose(transforms)

    im_dataset = ImageList(image_list, args.blur, args.bw, transform=transforms, imsize=args.imsize)

    print("loading model")
    net = load_model(args.model, args.checkpoint, args.GeM_p)
    net = net.cuda()

    print("computing features")

    
    with torch.no_grad():
        dataloader = torch.utils.data.DataLoader(
            im_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers
        )
        all_desc = []
        count = 0
        for i, x in enumerate(dataloader):
            t0 = time.time()
            
            try:
                features, _ = net.module.base(x.cuda())
            except:
                features = net.module.base(x.cuda())
                
            features = features.view(features.size(0), -1)
            
            features = net.module.projector_feat_bn(features)
            features = net.module.projector_feat_bn_1(features)
            
            all_desc.append(features)
            if (i + 1) % 10 == 0:
                print('Extract Features: [{}/{}]\t'.format(i + 1, len(dataloader)))
            
            if(len(all_desc)==100):
                all_desc = torch.vstack(tuple(all_desc)).cpu().numpy()
                t1 = time.time()
                print()
                print(f"image_description_time: {(t1 - t0)/args.batch_size:.5f} s per image")
                if args.pca_file:
                    print("Load PCA matrix", args.pca_file)
                    pca = faiss.read_VectorTransform(args.pca_file)
                    print(f"Apply PCA {pca.d_in} -> {pca.d_out}")
                    all_desc = pca.apply_py(all_desc)
                print("normalizing descriptors")
                faiss.normalize_L2(all_desc)
                name = args.o.split('/')[-1]
                names = name.split('_')
                out  = '/'.join(args.o.split('/')[:-1]) + '/' + names[0] + '_' + str(count) + '_' + '_'.join(names[1:])
                if not args.train_pca:
                    print(f"writing descriptors to {out}")
                    write_hdf5_descriptors(all_desc, image_list[count*args.batch_size*100:(count+1)*args.batch_size*100], out)
                count = count + 1
                all_desc = []
    
    if(len(all_desc) != 0):
        all_desc = torch.vstack(tuple(all_desc)).cpu().numpy()
        t1 = time.time()
        print()
        if args.train_pca:
            d = all_desc.shape[1]
            pca = faiss.PCAMatrix(d, args.pca_dim, -0.5)
            print(f"Train PCA {pca.d_in} -> {pca.d_out}")
            pca.train(all_desc)
            print(f"Storing PCA to {args.pca_file}")
            faiss.write_VectorTransform(pca, args.pca_file)
        elif args.pca_file:
            print("Load PCA matrix", args.pca_file)
            pca = faiss.read_VectorTransform(args.pca_file)
            print(f"Apply PCA {pca.d_in} -> {pca.d_out}")
            all_desc = pca.apply_py(all_desc)

        print("normalizing descriptors")
        faiss.normalize_L2(all_desc)
        name = args.o.split('/')[-1]
        names = name.split('_')
        out  = '/'.join(args.o.split('/')[:-1]) + '/' + names[0] + '_' + str(count) + '_' + '_'.join(names[1:])
        if not args.train_pca:
            print(f"writing descriptors to {out}")
            write_hdf5_descriptors(all_desc, image_list[count*args.batch_size*100:], out)

if __name__ == "__main__":
    main()