# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 14:40:21 2022

Feature extraction with timm models, and classification usgin sklearn

@author: scabini
"""

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Experimenter: compares models with different initialization")
    #data, paths, and other settings of general setup
    parser.add_argument('--data_path', type=str, default= '/home/DATA/datasets', help='Path to load/save torchvision or custom datasets')
    parser.add_argument('--output_path', type=str, default= '/home/DATA/results/DNN_features', help='Path for saving each models performance')    
    parser.add_argument('--gpu', type=str, default='0', help='GPU ID or list of ids ID1,ID2,... (see nvidia-smi), wrong ID will make the script use the CPU')
    parser.add_argument('--multigpu',  action='store_true', default=False, help='Either to use parallel GPU processing or not')

    #methods
    parser.add_argument('--model', type=str, default='resnet18', help='Name of an architecture to experiment with (see models.py')
    parser.add_argument('--depth', type=str, default='last', help='Depth of the feature map to use (last, middle, or quarter)')
    parser.add_argument('--pooling', type=str, default='AvgPool2d', help='Pooling technique (pytorch module/layer name, eg. AvgPool2d)')
    
    parser.add_argument('--dataset', type=str, default='LeavesTex1200', help='dataset name, same as the dataloader name')
    parser.add_argument('--grayscale',  action='store_true', default=False, help='Converts images to grayscale')
    parser.add_argument('--K', type=int, default=10, help='Number of splits for K-fold stratified cross validation')

    parser.add_argument('--input_dimm', type=int, default=224, help='Image input size (single value, square). The standard is a forced resize to 224 (square)')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size, increase for better speed, if you have enough VRAM')


    #hyperparameters
    # WIP
    parser.add_argument('--iterations', type=int, default=1, help='Number of random repetitions for k-fold/classifier seeds. Final results will consider average/variance of all K*iterations')
    parser.add_argument('--seed', type=int, default=6699, help='Base random seed for weight initialization and data splits/shuffle')
   
    return parser.parse_args()

import os
import ntpath
import numpy as np
import random
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
import multiprocessing
total_cores=multiprocessing.cpu_count()
os.environ["OMP_NUM_THREADS"]=str(total_cores*2)
args = parse_args()
# if not args.multigpu:
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
import torch
import torchvision
from feature_extraction import extract_features
import sklearn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn import model_selection
import datasets
import pickle
# from scipy import io
from datasets import getListOfFiles
from scipy import io

if __name__ == "__main__":
    args = parse_args()       
    
    DATASETS_ = {'DTD' : torchvision.datasets.DTD,
                 'FMD' : datasets.FMD,
                 'USPtex': datasets.USPtex,
                 'LeavesTex1200': datasets.LeavesTex1200,
                 'MBT': datasets.MBT,
                 'KTH-TIPS2-b': datasets.KTH_TIPS2_b,
                 'Outex' : datasets.Outex
                }

    ##########
    path = os.path.join(args.data_path, args.dataset)
    os.makedirs(os.path.join(args.output_path, 'feature_matrix', args.dataset), exist_ok=True)
    os.makedirs(os.path.join(args.output_path, 'classification', args.dataset), exist_ok=True)
     
    gtruth_= []
    preds_KNN, preds_LDA, preds_SVM = [], [], []
    accs_KNN, accs_LDA, accs_SVM = [], [], []
 
    print(args.model,  args.depth,  args.pooling, args.dataset, args.input_dimm)
    # print('Evaluating fold (...)= ', sep=' ', end='', flush=True)    
    file2 = [args.output_path +  '/classification/' + args.dataset + '/' + args.model + '_' + args.depth + '_' + args.pooling + '_'
                + args.dataset + '_' + str(args.input_dimm) +  '_gray' + str(args.grayscale) + '_K' + str(args.K) +'_EVALUATION.pkl'][0]

    base_seed = args.seed
    for it_ in range(args.iterations):
        seed = base_seed*(it_+1)
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        
        ### leave-one-out
        # splits = 1200
        # crossval = model_selection.KFold(n_splits=splits, shuffle=True, random_state=args.seed+1)
        
        ### kfold
        if args.dataset == 'Outex_13' or args.dataset == 'Outex_14':
            args.K = 1 #these Outexes have a single train/test split
        elif args.dataset == 'DTD':
            args.K = 10 #DTD have a fixed set of 10 splits
        elif 'KTH' in args.dataset:
            args.K = 4 #KTH2 have a fixed set of 4 splits
            
        splits = args.K #datasets with no defined sets get randomly K-splited
        
        if 'KTH' in args.dataset:
            crossval = model_selection.PredefinedSplit        
        elif args.dataset != 'DTD' and 'Outex' not in args.dataset:
            crossval = model_selection.StratifiedKFold(n_splits=splits, shuffle=True, random_state=seed+1)

        for partition in range(splits):
            # print(str(partition+1) + ', ', sep=' ', end='', flush=True)
            
            file = [args.output_path + '/feature_matrix/' + args.dataset + '/' + args.model + '_' + args.depth + '_' + args.pooling + '_'
                        + args.dataset + '_' + str(args.input_dimm) + '_gray' + str(args.grayscale) + '_split' +str(partition+1) + '.pkl'][0]
            
            if not os.path.isfile(file2):         
                if os.path.isfile(file) :
                    if args.dataset == 'DTD' or 'Outex' in args.dataset:
                        with open(file, 'rb') as f:
                            X_train,Y_train,X_test,Y_test = pickle.load(f) 
                    else:
                        if partition == 0:
                            with open(file, 'rb') as f:
                                X_, Y_, files = pickle.load(f)
                                # X_test,Y_test = X_train,Y_train
                               
                            
                            if 'KTH' in args.dataset:
                                crossval = crossval(DATASETS_[args.dataset](root=path, load_all=False).splits) 
                                
                            crossval.get_n_splits(X_, Y_)                    
                            crossval=crossval.split(X_, Y_) 
                            
                            
                        if 'KTH' in args.dataset: 
                            test_index, train_index = next(crossval)                    
                        else:
                            train_index, test_index = next(crossval) 
                            
                        X_test,Y_test = X_[test_index], Y_[test_index]
                        X_train, Y_train = X_[train_index], Y_[train_index]
                else:            
                    if args.dataset == 'DTD':
                        #Create a lambda function to pass data_transforms latter inside "extract_features"
                        dataset= lambda _transform: DATASETS_[args.dataset](root=path, download=True, split='train', 
                                                              partition=partition+1,
                                                              transform=_transform) 
                        
                        X_train, Y_train = extract_features(args.model, dataset, input_dimm=args.input_dimm, depth=args.depth, pooling=args.pooling,
                                                           batch_size=args.batch_size, multigpu=args.multigpu)        
                                      
                        dataset= lambda _transform: DATASETS_[args.dataset](root=path, download=True, split='test', 
                                                              partition = partition+1,
                                                              transform=_transform)
                        
                        X_test,Y_test = extract_features(args.model, dataset, input_dimm=args.input_dimm, depth=args.depth, pooling=args.pooling,
                                                         batch_size=args.batch_size, multigpu=args.multigpu)
                        # with open(file, 'wb') as f:
                        #     pickle.dump([X_train, Y_train, X_test, Y_test], f)
                        
                    elif 'Outex' in args.dataset:
                        outex_path = os.path.join(args.data_path, 'Outex')
                        dataset= lambda _transform: DATASETS_['Outex'](root=outex_path, split='train',
                                                              suite=args.dataset.split('_')[1],                                                      
                                                              transform=_transform) 
                        
                        X_train,Y_train = extract_features(args.model, dataset, input_dimm=args.input_dimm, depth=args.depth, pooling=args.pooling,
                                                           batch_size=args.batch_size, multigpu=args.multigpu)        
                        
                        dataset= lambda _transform: DATASETS_['Outex'](root=outex_path, split='test',
                                                              suite=args.dataset.split('_')[1],
                                                              transform=_transform)
                        
                        X_test,Y_test = extract_features(args.model, dataset, input_dimm=args.input_dimm, depth=args.depth, pooling=args.pooling,
                                                         batch_size=args.batch_size, multigpu=args.multigpu)
                        # with open(file, 'wb') as f:
                        #     pickle.dump([X_train, Y_train, X_test, Y_test], f)
                        
                    else:
                        if partition == 0:
                            files=getListOfFiles(path)
                            files = [ntpath.basename(f) for f in files]
                            dataset= lambda _transform: DATASETS_[args.dataset](root=path,
                                                                                transform= _transform, grayscale=args.grayscale)
                            
                            X_,Y_ = extract_features(args.model, dataset, input_dimm=args.input_dimm, depth=args.depth, pooling=args.pooling,
                                                     batch_size=args.batch_size, multigpu=args.multigpu) 
                            
                            # with open(file, 'wb') as f:
                            #     pickle.dump([X_, Y_, files], f)
                                
                            #we dont need this anymore, matlab is retired...
                            #if needed, a diferent script can be used to convert the .pkl files to .mat
                            # io.savemat(file + '.mat', {'X': X_, 'Y': Y_, 'files':files})
                            
                            # files = getListOfFiles(os.path.join(path, 'image'))
                            # files = [file for file in files if not file.endswith('.asv') 
                            #               and not file.endswith('.m')
                            #               and not file.endswith('.db')]
                            
                            # io.savemat('FMD_file_order.mat', {'files':files}) 
                            
                            
                            if 'KTH' in args.dataset:
                                crossval = crossval(DATASETS_[args.dataset](root=path, load_all=False).splits) 
                                
                            crossval.get_n_splits(X_, Y_)
                            crossval=crossval.split(X_, Y_)    
                            
                            
                        if 'KTH' in args.dataset: 
                            test_index, train_index = next(crossval)                    
                        else:
                            train_index, test_index = next(crossval)  
                            
                        X_test,Y_test = X_[test_index], Y_[test_index]
                        X_train, Y_train = X_[train_index], Y_[train_index]
             
    
            if not os.path.isfile(file2):  
                print(X_train.shape, X_test.shape, np.mean(X_train), np.mean(X_test))
                gtruth_.append(Y_test)
                
                KNN = KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto', 
                                           leaf_size=30, p=2, metric='minkowski', metric_params=None)
                
                KNN.fit(X_train,Y_train)
                preds=KNN.predict(X_test)              
                preds_KNN.append(preds)            
                acc= sklearn.metrics.accuracy_score(Y_test, preds)        
                accs_KNN.append(acc*100)     
                
                LDA= LinearDiscriminantAnalysis(solver='svd', 
                                                shrinkage=None, priors=None,
                                                n_components=None, 
                                                store_covariance=False, 
                                                tol=0.0001, covariance_estimator=None)        
                LDA.fit(X_train,Y_train)
                preds=LDA.predict(X_test)              
                preds_LDA.append(preds)            
                acc= sklearn.metrics.accuracy_score(Y_test, preds)
                accs_LDA.append(acc*100)  
        
                SVM = svm.SVC(C=1.0, kernel='rbf', degree=3, gamma='scale', 
                              coef0=0.0, shrinking=True, probability=False, tol=0.001,
                              cache_size=200, class_weight=None, verbose=False, 
                              max_iter=-1, decision_function_shape='ovr', 
                              break_ties=False, random_state=seed*(partition+1))
                
                SVM.fit(X_train,Y_train)
                preds=SVM.predict(X_test)            
                preds_SVM.append(preds)            
                acc= sklearn.metrics.accuracy_score(Y_test, preds)
                accs_SVM.append(acc*100)       

            results = {'gtruth_':gtruth_,
                        'preds_KNN':preds_KNN,
                        'accs_KNN':accs_KNN,
                        'preds_LDA':preds_LDA,
                        'accs_LDA':accs_LDA,
                        'preds_SVM':preds_SVM,
                        'accs_SVM':accs_SVM}  
         
# if not os.path.isfile(file2): 
#     with open(file2, 'wb') as f:
#         pickle.dump(results, f)
 
# if os.path.isfile(file2):           
#     with open(file2, 'rb') as f:
#         results = pickle.load(f) 
    
print('Classification results: ', sep=' ', end='', flush=True)   
print('KNN:', f"{np.round(np.mean(results['accs_KNN']), 2):.2f} (+-{np.round(np.std(results['accs_KNN']), 2):.2f})", sep=' ', end='', flush=True)      
print(' || LDA:', f"{np.round(np.mean(results['accs_LDA']), 2):.2f} (+-{np.round(np.std(results['accs_LDA']), 2):.2f})", sep=' ', end='', flush=True)      
print(' || SVM:', f"{np.round(np.mean(results['accs_SVM']), 2):.2f} (+-{np.round(np.std(results['accs_SVM']), 2):.2f})", sep=' ', end='', flush=True)      
print('\n', '-' * 70)
# print('\n#### FINAL METRICS ###')  
# print(args.model, args.dataset)      
# print('KNN:', f"{np.round(np.mean(results['accs_KNN']), 2):.2f} (+-{np.round(np.std(results['accs_KNN']), 2):.2f})")
# print('LDA:', f"{np.round(np.mean(results['accs_LDA']), 2):.2f} (+-{np.round(np.std(results['accs_LDA']), 2):.2f})")
# print('SVM:', f"{np.round(np.mean(results['accs_SVM']), 2):.2f} (+-{np.round(np.std(results['accs_SVM']), 2):.2f})")

        
### LATEX OUTPUT
# print(f"{np.round(np.mean(results['accs_KNN']), 2):.2f} ($\pm${np.round(np.std(results['accs_KNN']), 2):.2f})&",
#       f"{np.round(np.mean(results['accs_LDA']), 2):.2f} ($\pm${np.round(np.std(results['accs_LDA']), 2):.2f})&",
#       f"{np.round(np.mean(results['accs_SVM']), 2):.2f} ($\pm${np.round(np.std(results['accs_SVM']), 2):.2f})\\\\")            
            
            