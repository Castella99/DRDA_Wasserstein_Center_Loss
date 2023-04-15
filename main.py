import torch 
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from scipy import io
import numpy as np
from train_val_model import train_val, make_dataloader, test_model
from sklearn.model_selection import train_test_split, StratifiedKFold
import os
import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("-e", "--epoch", type=int, default=1000)
parser.add_argument("-p", "--patience", type=int, default=50)
parser.add_argument("-b", "--batch", type=int, default=64)
parser.add_argument("-l", "--Lambda", type=float, default=10)
parser.add_argument("-m", "--Mu", type=float)
parser.add_argument("-n", "--n", type=int, default=5)
parser.add_argument("-k", "--k_fold", type=int, default=5)
parser.add_argument("-t", "--test", type=int)

args = parser.parse_args()

print("-"*50)
print("Argument")
print("Epoch :", args.epoch)
print("Patience :", args.patience)
print("Batch size :", args.batch)
print("Lambda :", args.Lambda)
print("Mu :", args.Mu)
print("N :", args.n)
print("Fold :", args.k_fold)
print("Test Size :", args.test)

epoch = args.epoch
patience = args.patience
batch_size = args.batch
hyper_lambda = args.Lambda
hyper_mu = args.Mu
n_critics = args.n
k_fold = args.k_fold
test_size = args.test

os.getcwd()
output_dir = f"./output/result_cv_5fold_{hyper_mu}_{test_size}"
try :
    os.mkdir(output_dir+"/loss")
    os.mkdir(output_dir+"/cm")
except Exception as e :
    pass
finally :
    pass

# path
path = r'./../data_preprocessed_matlab/'  # 경로는 저장 파일 경로
file_list = os.listdir(path)

print("-"*50)
print("data path check")
for i in file_list:    # 확인
    print(i, end=' ')

Data = []
VAL = []
ARO = []

for i in tqdm(file_list, desc="read data"): 
    mat_file = io.loadmat(path+i)
    data = mat_file['data']
    labels = np.array(mat_file['labels'])
    val = labels.T[0]
    val = np.where(val<4, 1, val)
    val = np.where(np.logical_and(val>=4, val<7), 2, val)
    val = np.where(val>=7, 3, val)
    val = val.astype(np.int8)
    
    aro = labels.T[1]
    
    aro = np.where(aro<4, 1, aro)
    aro = np.where(np.logical_and(aro>=4, aro<7), 2, aro)
    aro = np.where(aro>=7, 3, aro)
    aro = aro.astype(np.int8)
    
    Data.append(data)
    VAL.append(val)
    ARO.append(aro)
        
Data = np.concatenate(Data,axis=0)   # 밑으로 쌓아서 하나로 만듬
VAL = np.concatenate(VAL,axis=0)
ARO = np.concatenate(ARO,axis=0)
print(Data.shape, VAL.shape, ARO.shape)

# eeg preprocessing

eeg_data = []
peripheral_data = []

for i in tqdm(range(len(Data)), desc="preprocess channel"):
    for j in range (40): 
        if(j < 32): # get channels 1 to 32
            eeg_data.append(Data[i][j])
        else:
            peripheral_data.append(Data[i][j])

# set data type, shape
eeg_data = np.reshape(eeg_data, (len(Data),1,32, 8064))
eeg_data = eeg_data.astype('float32')
eeg_data32 = torch.from_numpy(eeg_data)
VAL = (torch.from_numpy(VAL)).type(torch.long)
ARO = (torch.from_numpy(ARO)).type(torch.long)

test_ratio = test_size / len(Data)

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
x_train, x_test, y_train, y_test = train_test_split(eeg_data32, VAL, test_size=test_ratio, random_state=42, shuffle=True, stratify=VAL)

i = 1
acc_VAL = []
pre_VAL = []
rec_VAL = []
f1_VAL = []
roauc_VAL = []

print("\nValence\n")
for train_index, test_index in kf.split(x_train, y_train) :
    print('-'*50)
    print(f"{i} fold\n")
    x_train_fold, y_train_fold = x_train[train_index], y_train[train_index]
    x_source, x_target, y_source, y_target = train_test_split(x_train_fold, y_train_fold, test_size=0.5, random_state=42, shuffle=True, stratify=y_train_fold)
    x_val, y_val = x_train[test_index], y_train[test_index]

    source_VAL, target_VAL, val_VAL, test_VAL = make_dataloader(x_source, y_source, x_target, y_target, x_val, y_val, x_test, y_test, batch_size)
    fe_VAL, dis_VAL, cls_VAL = train_val(source_VAL, target_VAL, val_VAL, 'VALENCE', epoch, hyper_lambda, hyper_mu, n_critics, patience, output_dir, fold=i)
    acc, pre, rec, f1, roauc = test_model(fe_VAL, cls_VAL, test_VAL, 'VALENCE', output_dir, i)
    
    acc_VAL.append(acc)
    pre_VAL.append(pre)
    rec_VAL.append(rec)
    f1_VAL.append(f1)
    roauc_VAL.append(roauc)
    
    i += 1
    if i == (k_fold+1) :
        break
 
print("-"*50)
print("Valence Test Result")
print("Mean Acc :", sum(acc_VAL)/len(acc_VAL))
print("Mean Pre :", sum(pre_VAL)/len(pre_VAL))
print("Mean Rec :", sum(rec_VAL)/len(rec_VAL))
print("Mean F1 :", sum(f1_VAL)/len(f1_VAL))
print("Mean Auc :", sum(roauc_VAL)/len(roauc_VAL))
print()

acc_VAL.append(sum(acc_VAL)/len(acc_VAL))
pre_VAL.append(sum(pre_VAL)/len(pre_VAL))
rec_VAL.append(sum(rec_VAL)/len(rec_VAL))
f1_VAL.append(sum(f1_VAL)/len(f1_VAL))
roauc_VAL.append(sum(roauc_VAL)/len(roauc_VAL))

x_train, x_test, y_train, y_test = train_test_split(eeg_data32, ARO, test_size=test_ratio, random_state=42, shuffle=True, stratify=ARO)

i = 1
acc_ARO = []
pre_ARO = []
rec_ARO = []
f1_ARO = []
roauc_ARO = []

print("\nArousal\n")
for train_index, test_index in kf.split(x_train, y_train) :
    print('-'*50)
    print(f"{i} fold\n")
    x_train_fold, y_train_fold = x_train[train_index], y_train[train_index]
    x_source, x_target, y_source, y_target = train_test_split(x_train_fold, y_train_fold, test_size=0.5, random_state=42, shuffle=True, stratify=y_train_fold)
    x_val, y_val = x_train[test_index], y_train[test_index]

    source_ARO, target_ARO, val_ARO, test_ARO = make_dataloader(x_source, y_source, x_target, y_target, x_val, y_val, x_test, y_test, batch_size)
    fe_ARO, dis_ARO, cls_ARO = train_val(source_ARO, target_ARO, val_ARO, 'AROUSAL', epoch, hyper_lambda, hyper_mu, n_critics, patience, output_dir, fold=i)
    acc, pre, rec, f1, roauc = test_model(fe_ARO, cls_ARO, test_ARO, 'AROUSAL', output_dir, i)
    
    acc_ARO.append(acc)
    pre_ARO.append(pre)
    rec_ARO.append(rec)
    f1_ARO.append(f1)
    roauc_ARO.append(roauc)
    
    i += 1
    if i == (k_fold+1) :
        break

print("-"*50)
print("Arousal Test Result")
print("Mean Acc :", sum(acc_ARO)/len(acc_ARO))
print("Mean Pre :", sum(pre_ARO)/len(pre_ARO))
print("Mean Rec :", sum(rec_ARO)/len(rec_ARO))
print("Mean F1 :", sum(f1_ARO)/len(f1_ARO))
print("Mean Auc :", sum(roauc_ARO)/len(roauc_ARO))

acc_ARO.append(sum(acc_ARO)/len(acc_ARO))
pre_ARO.append(sum(pre_ARO)/len(pre_ARO))
rec_ARO.append(sum(rec_ARO)/len(rec_ARO))
f1_ARO.append(sum(f1_ARO)/len(f1_ARO))
roauc_ARO.append(sum(roauc_ARO)/len(roauc_ARO))

result_dict = {"Acc_VAL" : acc_VAL, "Pre_VAL" : pre_VAL, "Rec_VAL" : rec_VAL, "F1_VAL" : f1_VAL, "Auc_VAL" : roauc_VAL, "Acc_ARO" : acc_ARO, "Pre_ARO" : pre_ARO, "Rec_ARO" : rec_ARO, "F1_ARO" : f1_ARO, "Auc_ARO" : roauc_ARO}

df = pd.DataFrame(data=result_dict, index=["1", "2", "3", "4", "5", "Mean"])
df.to_csv(output_dir+'/result_table.csv')