# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 11:52:22 2022

Personal dataloaders for texture datasets

@author: scabini
"""
import os
import ntpath
import torch
from torchvision import datasets
from torchvision.datasets import VisionDataset as Dataset
from torchvision.datasets.utils import verify_str_arg, download_and_extract_archive
from typing import Optional, Callable
import pathlib
from PIL import Image
import numpy as np

#this is for getting all images in a directory (including subdirs)
def getListOfFiles(dirName):
    # create a list of all files in a root dir
    listOfFile = os.listdir(dirName)
    allFiles = list()
    for entry in listOfFile:
        fullPath = os.path.join(dirName, entry)
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)
                
    return allFiles

class Vistex(Dataset):
    """Vistex    
    """    
    def __init__(self, root, transform=None, load_all=True, grayscale=False):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.grayscale=grayscale
        self._image_files = getListOfFiles(root)
        self.transform = transform
        self.load_all=load_all
        self.data = []
        self.targets = []
        if self.load_all:
            for img_name in self._image_files:
                if self.grayscale:
                    self.data.append(Image.open(img_name).convert('L').convert('RGB'))
                else:
                    self.data.append(Image.open(img_name).convert('RGB'))                     
                self.targets.append(int(ntpath.basename(img_name).split('_')[0][1:]))
        print(np.unique(self.targets), 'classes')
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()        
        if self.load_all:
            image = self.data[idx]
            target = self.targets[idx]
        else:
            img_name = self._image_files[idx]
            if self.grayscale:
                image = Image.open(img_name).convert('L').convert('RGB')
            else:
                image = Image.open(img_name).convert('RGB')                
            target = int(ntpath.basename(img_name).split('_')[0][1:])        
        if self.transform:
            image = self.transform(image)
        return image, target


class CURet(Dataset):
    """CURet    
    """    
    def __init__(self, root, transform=None, load_all=True, grayscale=False):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.grayscale=grayscale
        self._image_files = getListOfFiles(root)
        self.transform = transform
        self.load_all=load_all
        self.data = []
        self.targets = []
        if self.load_all:
            for img_name in self._image_files:
                if self.grayscale:
                    self.data.append(Image.open(img_name).convert('L').convert('RGB'))
                else:
                    self.data.append(Image.open(img_name).convert('RGB'))                     
                self.targets.append(int(ntpath.basename(img_name).split('_')[0]))
        print(np.unique(self.targets), 'classes')

    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()        
        if self.load_all:
            image = self.data[idx]
            target = self.targets[idx]
        else:
            img_name = self._image_files[idx]
            if self.grayscale:
                image = Image.open(img_name).convert('L').convert('RGB')
            else:
                image = Image.open(img_name).convert('RGB')                
            target = int(ntpath.basename(img_name).split('_')[0])        
        if self.transform:
            image = self.transform(image)
        return image, target



class USPtex(Dataset):
    """USPtex - natural texture dataset
    Backes, André Ricardo, Dalcimar Casanova, and Odemir Martinez Bruno. 
    "Color texture analysis based on fractal descriptors." 
    Pattern Recognition (2012)  
    http://scg-turing.ifsc.usp.br/data/bases/USPtex.zip
    """    
    def __init__(self, root, transform=None, load_all=True, grayscale=False):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.grayscale=grayscale
        self._image_files = getListOfFiles(root)
        self.transform = transform
        self.load_all=load_all
        self.data = []
        self.targets = []
        if self.load_all:
            for img_name in self._image_files:
                if self.grayscale:
                    self.data.append(Image.open(img_name).convert('L').convert('RGB'))
                else:
                    self.data.append(Image.open(img_name).convert('RGB'))  
                    
                self.targets.append(int(ntpath.basename(img_name).split('_')[0][1:]))
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        if self.load_all:
            image = self.data[idx]
            target = self.targets[idx]
        else:
            img_name = self._image_files[idx]
            if self.grayscale:
                image = Image.open(img_name).convert('L').convert('RGB')
            else:
                image = Image.open(img_name).convert('RGB') 
                
            target = int(ntpath.basename(img_name).split('_')[0][1:])
        
        if self.transform:
            image = self.transform(image)

        return image, target
   
      
class LeavesTex1200(Dataset):
    """1200tex - leaf textures
    Casanova, Dalcimar, Jarbas Joaci de Mesquita Sá Junior, and Odemir Martinez Bruno.
    "Plant leaf identification using Gabor wavelets."
    International Journal of Imaging Systems and Technology (2009) 
    http://scg-turing.ifsc.usp.br/data/bases/LeavesTex1200.zip
    """
    def __init__(self, root, transform=None, load_all=True, grayscale=False):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.grayscale=grayscale
        self._image_files = getListOfFiles(root)
        self.transform = transform
        self.load_all=load_all
        self.data = []
        self.targets = []
        if self.load_all:
            for img_name in self._image_files:
                if self.grayscale:
                    self.data.append(Image.open(img_name).convert('L').convert('RGB'))
                else:
                    self.data.append(Image.open(img_name).convert('RGB'))  
                self.targets.append(int(ntpath.basename(img_name).split('_')[0][1:]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()        
        if self.load_all:
            image = self.data[idx]
            target = self.targets[idx]
        else:
            img_name = self._image_files[idx]
            if self.grayscale:
                image = Image.open(img_name).convert('L').convert('RGB')
            else:
                image = Image.open(img_name).convert('RGB')
            target = int(ntpath.basename(img_name).split('_')[0][1:])
        
        if self.transform:
            image = self.transform(image)

        return image, target


class MBT(Dataset):
    """Multi Band Texture (MBT)
    S. Abdelmounaime, Dong-Chen H.
    New brodatz-based image databases for grayscale color and multiband texture analysis
    ISRN Mach. Vis., 2013 (2013)
    
    The splits here follows the approach (16 non-overlaping crops) described in: 
    
    Scabini, Leonardo FS, et al.
    "Multilayer complex network descriptors for color–texture characterization."
    Information Sciences 491 (2019): 30-47.
    """    
    def __init__(self, root, transform=None, load_all=True, grayscale=False):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.grayscale=grayscale
        self._image_files = getListOfFiles(root)
        self.transform = transform
        self.load_all=load_all
        self.data = []
        self.targets = []
        if self.load_all:
            for img_name in self._image_files:
                if self.grayscale:
                    self.data.append(Image.open(img_name).convert('L').convert('RGB'))
                else:
                    self.data.append(Image.open(img_name).convert('RGB')) 
                self.targets.append(int(ntpath.basename(img_name).split('_')[0][2:]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        if self.load_all:
            image = self.data[idx]
            target = self.targets[idx]
        else:
            img_name = self._image_files[idx]
            if self.grayscale:
                image = Image.open(img_name).convert('L').convert('RGB')
            else:
                image = Image.open(img_name).convert('RGB') 
            target = int(ntpath.basename(img_name).split('_')[0][2:])
        
        if self.transform:
            image = self.transform(image)

        return image, target
 
    
class KTH_TIPS2_b(Dataset):
     """KTH-TIPS2-b
     Caputo, Barbara, Eric Hayman, and P. Mallikarjuna.
     "Class-specific material categorisation."
     Tenth IEEE International Conference on Computer Vision (ICCV'05)
     Volume 1. Vol. 2. IEEE, 2005.
     https://www.csc.kth.se/cvap/databases/kth-tips/kth-tips2-b_col_200x200.tar
     
     Validation splits: divided by samples (4-folds)
     """     
     def __init__(self, root, transform=None, load_all=True, grayscale=False):
         """
         Args:
             root_dir (string): Directory with the original dataset.
             transform (callable, optional): Optional transform to be applied
                 on a sample.
             load_all: either to open all images in memory, or one by one later
             grasycale: either to use grayscale images or not
         """
         self.class_names = {'aluminium_foil':0, 'brown_bread':1, 'corduroy':2,
                             'cork':3, 'cotton':4, 'cracker':5, 'lettuce_leaf':6,
                             'linen':7, 'white_bread':8, 'wood':9, 'wool':10}
         
         self.split_names = {'sample_a':0, 'sample_b':1, 'sample_c':2, 'sample_d':3}
         
         self.grayscale = grayscale         
         self._image_files = getListOfFiles(root)         
         self._image_files = [file for file in self._image_files if not file.endswith('.txt') and not file.endswith('.pdf')]
         
         self.transform = transform
         self.load_all=load_all
         self.data = []
         self.targets = []
         self.splits = [self.split_names[img_name.split('/')[-2]] for img_name in self._image_files]
         if self.load_all:
             for img_name in self._image_files:
                 if self.grayscale:
                     self.data.append(Image.open(img_name).convert('L').convert('RGB'))
                 else:
                     self.data.append(Image.open(img_name).convert('RGB'))                     
             
                 self.targets.append(self.class_names[img_name.split('/')[-3]])
                 

     def __len__(self):
         return len(self.data)

     def __getitem__(self, idx):
         if torch.is_tensor(idx):
             idx = idx.tolist()         
         if self.load_all:
             image = self.data[idx]
             target = self.targets[idx]
         else:
             img_name = self._image_files[idx]             
             if self.grayscale:
                 image = Image.open(img_name).convert('L').convert('RGB')
             else:
                 image = Image.open(img_name).convert('RGB')              
             target = self.class_names[img_name.split('/')[-3]]
         
         if self.transform:
             image = self.transform(image)

         return image, target   
 
    
class FMD(Dataset):
     """Flickr Material Dataset (FMD)
     Sharan, Lavanya, Ruth Rosenholtz, and Edward Adelson.
     "Material perception: What can you see in a brief glance?."
     Journal of Vision 9.8 (2009): 784-784.   
     http://people.csail.mit.edu/celiu/CVPR2010/FMD/FMD.zip
     """     
     def __init__(self, root, transform=None, load_all=True, grayscale=False):
         """
         Args:
             root_dir (string): Directory with the original dataset.
             transform (callable, optional): Optional transform to be applied
                 on a sample.
             load_all: either to open all images in memory, or one by one later
             grasycale: either to use grayscale images or not
         """
         self.grayscale = grayscale         
         self._image_files = getListOfFiles(os.path.join(root, 'image'))
         self._image_files = [file for file in self._image_files if not file.endswith('.asv') 
                       and not file.endswith('.m')
                       and not file.endswith('.db')]
         self.class_names = {'fabric':0, 'foliage':1, 'glass':2, 'leather':3, 'metal':4,
                             'paper':5, 'plastic':6, 'stone':7, 'water':8, 'wood':9}
         self.transform = transform
         self.load_all=load_all
         self.data = []
         self.targets = []
         if self.load_all:
             for img_name in self._image_files:
                 if self.grayscale:
                     self.data.append(Image.open(img_name).convert('L').convert('RGB'))
                 else:
                     self.data.append(Image.open(img_name).convert('RGB'))                     
                 
                 self.targets.append(self.class_names[img_name.split('/')[-2]])

     def __len__(self):
         return len(self.data)

     def __getitem__(self, idx):
         if torch.is_tensor(idx):
             idx = idx.tolist()
         
         if self.load_all:
             image = self.data[idx]
             target = self.targets[idx]
         else:
             img_name = self._image_files[idx]
             
             if self.grayscale:
                 image = Image.open(img_name).convert('L').convert('RGB')
             else:
                 image = Image.open(img_name).convert('RGB')             
             
             target = self.class_names[img_name.split('/')[-2]]
         
         if self.transform:
             image = self.transform(image)

         return image, target  
    
        
class Outex(Dataset):
      """ Outex   
      Ojala, Timo, et al.
      "Outex-new framework for empirical evaluation of texture analysis algorithms."
      2002 International Conference on Pattern Recognition. Vol. 1. IEEE, 2002.
      http://scg-turing.ifsc.usp.br/data/bases/Outex.zip
      """     
      def __init__(self, root, split, suite='13', transform=None, load_all=True, grayscale=False):
          """
          Args:
              root_dir (string): Directory with the original dataset.
              suite (string): The outex test suite, from 00, 01, ... to 16
              split (string): 'train' or 'test'
              transform (callable, optional): Optional transform to be applied
                  on a sample.
              load_all: either to open all images in memory, or one by one later
              grasycale: either to use grayscale images or not
          """
          self.suite = 'Outex_TC_000' + suite
          self.grayscale = grayscale     
          self.transform = transform
          self.load_all=load_all
          self.data = [] 
          self._image_files = []
          self.targets = [] 
                    
          with open(root + '/' + self.suite + '/000/' + split +'.txt','r') as f:
              lines = f.readlines()
              f.close()
          lines.pop(0)
          for item in lines:
              self._image_files.append(root + '/' + self.suite + '/images/' + item.split(' ')[0])
              self.targets.append(int(item.split(' ')[1]))
   
          if self.load_all:
              for img_name in self._image_files:                  
                if self.grayscale:
                    self.data.append(Image.open(img_name).convert('L').convert('RGB'))
                else:
                    self.data.append(Image.open(img_name).convert('RGB'))                  

      def __len__(self):
          return len(self.data)

      def __getitem__(self, idx):
          if torch.is_tensor(idx):
              idx = idx.tolist()
          
          if self.load_all:
              image = self.data[idx]
              
          else:
              img_name = self._image_files[idx]
              
              if self.grayscale:
                  image = Image.open(img_name).convert('L').convert('RGB')
              else:
                  image = Image.open(img_name).convert('RGB')   
                  
          target = self.targets[idx]
          
          if self.transform:
              image = self.transform(image)

          return image, target
      
class DTD(Dataset):
    """`Describable Textures Dataset (DTD) <https://www.robots.ox.ac.uk/~vgg/data/dtd/>`_.

    Args:
        root (string): Root directory of the dataset.
        split (string, optional): The dataset split, supports ``"train"`` (default), ``"val"``, or ``"test"``.
        partition (int, optional): The dataset partition. Should be ``1 <= partition <= 10``. Defaults to ``1``.

            .. note::

                The partition only changes which split each image belongs to. Thus, regardless of the selected
                partition, combining all splits will result in all images.

        transform (callable, optional): A function/transform that  takes in a PIL image and returns a transformed
            version. E.g, ``transforms.RandomCrop``.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
        download (bool, optional): If True, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again. Default is False.
    """

    _URL = "https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz"
    _MD5 = "fff73e5086ae6bdbea199a49dfb8a4c1"

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:

        super().__init__(root, transform=transform, target_transform=target_transform)
        self._base_folder = pathlib.Path(self.root) / type(self).__name__.lower()
        self._data_folder = self._base_folder / "dtd"
        self._meta_folder = self._data_folder / "labels"
        self._images_folder = self._data_folder / "images"

        if download:
            self._download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")

        self._image_files = []
        classes = []
        self.names = []
        with open(self._meta_folder / "labels_joint_anno.txt") as file:
            for line in file:
                strs = line.strip().split("/")
                cls= strs[0]
                name = strs[1].strip().split(" ")[0]
                self._image_files.append(self._images_folder.joinpath(cls, name))
                self.names.append(name)
                classes.append(cls)

        self.name_to_idx = dict(zip(self.names, range(len(self.names))))
        self.classes = sorted(set(classes))
        self.class_to_idx = dict(zip(self.classes, range(len(self.classes))))
        self._labels = [self.class_to_idx[cls] for cls in classes]

    def __len__(self) -> int:
        return len(self._image_files)

    def __getitem__(self, idx):
        image_file, label = self._image_files[idx], self._labels[idx]
        image = Image.open(image_file).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label
    
    def get_indexes(self,split="train", partition=1):
        
        verify_str_arg(split, "split", ("train", "val", "test"))
        if not isinstance(partition, int) and not (1 <= partition <= 10):
            raise ValueError(
                f"Parameter 'partition' should be an integer with `1 <= partition <= 10`, "
                f"but got {partition} instead"
            )
        
        indexes = []
        
        with open(self._meta_folder / f"{split}{partition}.txt") as file:
            for line in file:
                name = line.strip().split("/")[1]
                indexes.append(self.name_to_idx[name])                
        
        return indexes

    def extra_repr(self) -> str:
        return f"split={self._split}, partition={self._partition}"

    def _check_exists(self) -> bool:
        return os.path.exists(self._data_folder) and os.path.isdir(self._data_folder)

    def _download(self) -> None:
        if self._check_exists():
            return
        download_and_extract_archive(self._URL, download_root=str(self._base_folder), md5=self._MD5)
        
class MINC(Dataset):  
    _URL = "http://opensurfaces.cs.cornell.edu/static/minc/minc-2500.tar.gz"
    
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        
        super().__init__(root, transform=transform, target_transform=target_transform)
        self._base_folder = pathlib.Path(self.root)
        self._data_folder = self._base_folder / "minc-2500"
        self._meta_folder = self._data_folder / "labels"
        self._images_folder = self._data_folder / "images"
        
        if download:
            self._download()
            
        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")
        
        self._image_files = []
        classes = []
        self.names = []
        with open(self._meta_folder / "train1.txt") as file:
            for line in file:
                _, cl, name = line.strip().split("/")
                self._image_files.append(self._images_folder.joinpath(cl, name))
                self.names.append(name)
                classes.append(cl)
        with open(self._meta_folder / "validate1.txt") as file:
            for line in file:
                _, cl, name = line.strip().split("/")
                self._image_files.append(self._images_folder.joinpath(cl, name))
                self.names.append(name)
                classes.append(cl)
        with open(self._meta_folder / "test1.txt") as file:
            for line in file:
                _, cl, name = line.strip().split("/")
                self._image_files.append(self._images_folder.joinpath(cl, name))
                self.names.append(name)
                classes.append(cl)
                
        self.name_to_idx = dict(zip(self.names, range(len(self.names))))
        self.classes = sorted(set(classes))
        self.class_to_idx = dict(zip(self.classes, range(len(self.classes))))
        self._labels = [self.class_to_idx[cl] for cl in classes]
        
    def __len__(self) -> int:
        return len(self._image_files)
    
    def __getitem__(self, idx):
        image_file, label = self._image_files[idx], self._labels[idx]
        image = Image.open(image_file).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        if self.target_transform:
            label = self.target_transform(label)
            
        return image, label
    
    def get_indexes(self, split="train", partition=1):
        
        verify_str_arg(split, "split", ("train", "validate", "test"))
        if not isinstance(partition, int) and not (1 <= partition <= 5):
            raise ValueError(
                f"Parameter 'partition' shoud be an integer with `1 <= partition <= 5`, "
                f"but got {partition} instead"
            )
            
        indexes = []
        
        with open(self._meta_folder / f"{split}{partition}.txt") as file:
            for line in file:
                name = line.strip().split("/")[2]
                indexes.append(self.name_to_idx[name])
                
        return indexes
    
    def extra_repr(self) -> str:
        return f"split={self.split}, partition={self.partition}"
    
    def _check_exists(self) -> bool:
        return os.path.exists(self._data_folder) and os.path.isdir(self._data_folder)
    
    def _download(self) -> None:
        if self._check_exists():
            return
        download_and_extract_archive(self._URL, download_root=str(self._base_folder))
        
def GTOS_mobile(root, split='train', transform=None):
    """
    need to fill meta info about GTOS-Mobile...
    """  
    if split == 'train':
        return datasets.ImageFolder(os.path.join(root, 'train'), transform)
    elif split == 'test':
        return datasets.ImageFolder(os.path.join(root, 'test'), transform)
    else:
        print("GTOS-MOBILE = WRONG SPLIT NAME!!!")
              


#%%
def find_classes(classdir):
    classes = []
    class_to_idx = {}
    with open(classdir, 'r') as f:
        for line in f:
            label, name = line.split(' ')
            classes.append(name)
            class_to_idx[name] = int(label) - 1
    return classes, class_to_idx

#%%
def make_dataset(txtname, datadir):
    rgbimages = []
    diffimages = []
    names = []
    labels = []
    classes = []
    with open(txtname / "train1.txt", "r") as lines:
        for line in lines:
            cl = line.split('/')[0]
            name, label = line.split(' ')
            name = name.split('/')[-1]
            for filename in os.listdir(os.path.join(datadir, 'color_imgs', name)):
                _rgbimg = os.path.join(datadir, 'color_imgs', name, filename)
                names.append(filename[:-4])
                _diffimg = os.path.join(datadir, 'diff_imgs', name, filename)
                assert os.path.isfile(_rgbimg)
                rgbimages.append(_rgbimg)
                diffimages.append(_diffimg)
                labels.append(int(label)-1)
                classes.append(cl) 
                
                
    with open(txtname /"test1.txt", "r") as lines:
        for line in lines:
            cl = line.split('/')[0]
            name, label = line.split(' ')
            name = name.split('/')[-1]
            for filename in os.listdir(os.path.join(datadir, 'color_imgs', name)):
                _rgbimg = os.path.join(datadir, 'color_imgs', name, filename)
                names.append(filename[:-4])
                _diffimg = os.path.join(datadir, 'diff_imgs', name, filename)
                assert os.path.isfile(_rgbimg)
                rgbimages.append(_rgbimg)
                diffimages.append(_diffimg)
                labels.append(int(label)-1) 
                classes.append(cl)

    return rgbimages, names, diffimages, labels, classes

#%%
class GTOS(Dataset):
    def __init__(self, root, transform=None, download=False):
        self._base_folder = pathlib.Path(root)
        self._data_folder = self._base_folder
        self._meta_folder = self._data_folder / "labels"
           
        self.transform = transform

        self.rgbimages, names, _, _, classes = make_dataset(self._meta_folder, self._data_folder)
        assert (len(self.rgbimages) == len(classes))
        
        self.name_to_idx = dict(zip(names, range(len(names))))
        self.classes = sorted(set(classes))
        self.class_to_idx = dict(zip(self.classes, range(len(self.classes))))
        self.labels = [self.class_to_idx[cl] for cl in classes]

    def get_indexes(self, split="train", partition=1):
        
        verify_str_arg(split, "split", ("train", "test"))
        if not isinstance(partition, int) and not (1 <= partition <= 5):
            raise ValueError(
                f"Parameter 'partition' shoud be an integer with `1 <= partition <= 5`, "
                f"but got {partition} instead"
            )
            
        indexes = []
        
        with open(self._meta_folder / f"{split}{partition}.txt") as file:
            for line in file:
                folder = line.strip().split("/")[1].split(" ")[0]
                for name in os.listdir(self._data_folder / "color_imgs" / folder):
                    indexes.append(self.name_to_idx[name[:-4]])
                
        return indexes

    def __getitem__(self, index):
        _rgbimg = Image.open(self.rgbimages[index]).convert('RGB')
        # _diffimg = Image.open(self.diffimages[index]).convert('RGB')
        _label = self.labels[index]
        if self.transform is not None:
            _rgbimg = self.transform(_rgbimg)

        return _rgbimg, _label

    def __len__(self):
        return len(self.rgbimages)
        # return 10000              
              
 #%%         
          
          
          
          
# %%
