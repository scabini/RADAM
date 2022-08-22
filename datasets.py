# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 11:52:22 2022

Personal dataloaders for texture datasets

@author: scabini
"""
import os
import ntpath
import torch
from torchvision.datasets import VisionDataset as Dataset
from PIL import Image

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

class USPtex(Dataset):
    """USPtex - natural texture dataset
    Backes, André Ricardo, Dalcimar Casanova, and Odemir Martinez Bruno. 
    "Color texture analysis based on fractal descriptors." 
    Pattern Recognition (2012)    
    """    
    def __init__(self, root, transform=None, load_all=True, grayscale=False):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.grayscale=grayscale
        self.files = getListOfFiles(root)
        self.transform = transform
        self.load_all=load_all
        self.data = []
        self.targets = []
        if self.load_all:
            for img_name in self.files:
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
            img_name = self.files[idx]
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
    """
    def __init__(self, root, transform=None, load_all=True, grayscale=False):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.grayscale=grayscale
        self.files = getListOfFiles(root)
        self.transform = transform
        self.load_all=load_all
        self.data = []
        self.targets = []
        if self.load_all:
            for img_name in self.files:
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
            img_name = self.files[idx]
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
        self.files = getListOfFiles(root)
        self.transform = transform
        self.load_all=load_all
        self.data = []
        self.targets = []
        if self.load_all:
            for img_name in self.files:
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
            img_name = self.files[idx]
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
     
     Validation splits: ?, some papers use random 10-fold, some use subfolders
                        as folds
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
         self.files = getListOfFiles(root)         
         self.files = [file for file in self.files if not file.endswith('.txt') and not file.endswith('.pdf')]
         
         self.transform = transform
         self.load_all=load_all
         self.data = []
         self.targets = []
         self.splits = [self.split_names[img_name.split('/')[-2]] for img_name in self.files]
         if self.load_all:
             for img_name in self.files:
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
             img_name = self.files[idx]
             
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
         self.files = getListOfFiles(os.path.join(root, 'image'))
         self.files = [file for file in self.files if not file.endswith('.asv') 
                       and not file.endswith('.m')
                       and not file.endswith('.db')]
         self.class_names = {'fabric':0, 'foliage':1, 'glass':2, 'leather':3, 'metal':4,
                             'paper':5, 'plastic':6, 'stone':7, 'water':8, 'wood':9}
         self.transform = transform
         self.load_all=load_all
         self.data = []
         self.targets = []
         if self.load_all:
             for img_name in self.files:
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
             img_name = self.files[idx]
             
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
          self.files = []
          self.targets = [] 
                    
          with open(root + '/' + self.suite + '/000/' + split +'.txt','r') as f:
              lines = f.readlines()
              f.close()
          lines.pop(0)
          for item in lines:
              self.files.append(root + '/' + self.suite + '/images/' + item.split(' ')[0])
              self.targets.append(int(item.split(' ')[1]))
   
          if self.load_all:
              for img_name in self.files:                  
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
              img_name = self.files[idx]
              
              if self.grayscale:
                  image = Image.open(img_name).convert('L').convert('RGB')
              else:
                  image = Image.open(img_name).convert('RGB')   
                  
          target = self.targets[idx]
          
          if self.transform:
              image = self.transform(image)

          return image, target