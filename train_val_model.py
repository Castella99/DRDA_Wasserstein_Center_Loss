import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from model import FE, Discriminator, Classifier, Wasserstein_Loss, CenterLoss
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, average_precision_score, f1_score, roc_auc_score, roc_curve, ConfusionMatrixDisplay, auc, RocCurveDisplay
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import copy
from tqdm import tqdm

def make_dataloader(x_source, y_source, x_target, y_target, x_val, y_val, x_test, y_test, batch_size=64) :
    # make data loader
    print("make data loader\n")
    target_dataset = TensorDataset(x_target, y_target)
    source_dataset = TensorDataset(x_source, y_source)
    val_dataset = TensorDataset(x_val, y_val)
    test_dataset = TensorDataset(x_test, y_test)
    target_dataloader = DataLoader(target_dataset, batch_size, shuffle=True)
    source_dataloader = DataLoader(source_dataset, batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size, shuffle=True)
    return source_dataloader, target_dataloader, val_dataloader, test_dataloader

def train_val(source_dataloader, target_dataloader, val_dataloader, label, nb_epochs, hyper_lambda, hyper_mu, hyper_n, patience, output_dir, fold=0) :
    print("-"*50)
    print(f"{label} label train and valiation")
    # cuda
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("device: ", device)
    
    #model
    dis = Discriminator().cuda()
    fe = FE().cuda()
    classifier = Classifier(cls_num=3).cuda()
    centerloss = CenterLoss(feat_dim=64, num_classes=3).cuda()

    #optim
    optimizer_dis = optim.Adam(dis.parameters(),lr=0.0001,betas=(0.5,0.999))
    optimizer_fe = optim.Adam(fe.parameters(),lr=0.0001, betas=(0.5,0.999))
    optimizer_cls = optim.Adam(classifier.parameters(),lr=0.0001)
    optimizer_centerloss = optim.SGD(centerloss.parameters(), lr=0.5)

    #cls_loss
    cls_loss = nn.CrossEntropyLoss().cuda()

    # train WGAN
    accuracy_s = []
    accuracy_t = []
    accuracy_val = []
    val_loss_list = []

    best_loss = 10000000
    best_acc = 0.0
    best_f1 = 0.0
    best_acc_f1 = 0.0
    limit_check = 0
    val_loss = 0
    best_val_epoch = 1
    best_acc_f1_epoch = 1

    torch.autograd.set_detect_anomaly(True)

    epochs = 0
    print()
    # while parameter converge
    for epoch in tqdm(range(nb_epochs)):
        temp_accuracy_t = 0
        temp_accuracy_s = 0
        temp_accuracy_val = 0
        temp_gloss = 0
        temp_wdloss = 0
        temp_gradloss = 0
        temp_clsloss = 0
        temp_centloss = 0

        print(epoch+1, ": epoch")

        temp = 0.0 #batch count
        fe.train()
        dis.train()
        classifier.train()
        centerloss.train()
        # batch
        for i, (target, source) in enumerate(zip(target_dataloader, source_dataloader)):
            temp += 1.0

            x_target = target[0].to(device)
            y_target = target[1].to(device)
            x_source = source[0].to(device)
            y_source = source[1].to(device)

            # update discriminator
            for p in fe.parameters() :
                p.requires_grad = False
            for p in dis.parameters() :
                p.requires_grad = True
            for p in classifier.parameters() :
                p.requires_grad = False
            for p in centerloss.parameters() :
                p.requires_grad = False
            
            for k in range(hyper_n) :
                optimizer_dis.zero_grad()
                
                feat_t = fe(x_target)
                feat_s = fe(x_source)
                
                dc_t = dis(feat_t)
                dc_s = dis(feat_s)
                
                epsil = torch.rand(1).item()
                feat = epsil*feat_s+(1-epsil)*feat_t
                feat.requires_grad_()
                dc = dis(feat)
                
                wd_loss = torch.mean(dc_t) - torch.mean(dc_s)
                grad = torch.autograd.grad(outputs=dc, inputs=feat, grad_outputs=torch.ones(dc.size()).cuda(), create_graph=True, retain_graph=True)[0]
                grad = grad.view(grad.shape[0], -1)
                grad_norm = grad.norm(2, dim=1)
                grad_pt = torch.mean((grad_norm-1)**2)
                wd_grad_loss = wd_loss + hyper_lambda*grad_pt
                wd_grad_loss.backward()
                optimizer_dis.step()

            # update classifier
            for p in fe.parameters() :
                p.requires_grad = False
            for p in dis.parameters() :
                p.requires_grad = False
            for p in classifier.parameters() :
                p.requires_grad = True
            for p in centerloss.parameters() :
                p.requires_grad = False
            
            optimizer_cls.zero_grad()
            feat_s = fe(x_source)
            pred_s = classifier(feat_s)
            cls_loss_source = cls_loss(pred_s, y_source-1)
            cls_loss_source.backward()
            optimizer_cls.step()
            
            # update Feature Extractor
            for p in fe.parameters() :
                p.requires_grad = True
            for p in dis.parameters() :
                p.requires_grad = False
            for p in classifier.parameters() :
                p.requires_grad = False
            for p in centerloss.parameters() :
                p.requires_grad = True
            
            optimizer_fe.zero_grad()
            optimizer_centerloss.zero_grad()
            feat_t = fe(x_target)
            feat_s = fe(x_source)
            pred_s = classifier(feat_s)
            dc_t = dis(feat_t)
            dc_s = dis(feat_s)
            wd_loss = torch.mean(dc_s) - torch.mean(dc_t)
            cls_loss_source = cls_loss(pred_s, y_source-1)
            center_loss = centerloss(feat_t, y_target)
            fe_loss = cls_loss_source + hyper_mu*wd_loss + 0.5*center_loss
            fe_loss.backward()
            for p in centerloss.parameters() :
                p.grad.data *= (1//0.5)
            optimizer_fe.step()
            optimizer_centerloss.step()
            
            feat_t = fe(x_target)
            feat_s = fe(x_source)
            pred_t = classifier(feat_t)
            pred_s = classifier(feat_s)

            # Temp_Loss
            wd_loss = Wasserstein_Loss(dc_s, dc_t).detach().cpu()
            cls_loss_source = cls_loss(pred_s, y_source-1).detach().cpu()
        
            center_loss = centerloss(feat_t, y_target).detach().cpu()
            g_loss = cls_loss_source + hyper_mu*wd_loss + 0.5*center_loss
            
            temp_wdloss = temp_wdloss + wd_loss
            temp_clsloss = temp_clsloss+ cls_loss_source
            temp_centloss = temp_centloss+ center_loss
            temp_gloss = temp_gloss+ g_loss

            temp_accuracy_t += ((torch.argmax(pred_t,1)+1)== y_target).to(torch.float).mean().detach().cpu()
            temp_accuracy_s += ((torch.argmax(pred_s,1)+1)== y_source).to(torch.float).mean().detach().cpu()
        
        print("\ngloss :", temp_gloss.item()/temp)
        print("wd_loss :", temp_wdloss.item()/temp)
        print("cls_loss :", temp_clsloss.item()/temp)
        print("center_loss :", temp_centloss.item()/temp)
        print("acc_t :", temp_accuracy_t.item()/temp)
        print("acc_s :", temp_accuracy_s.item()/temp)
        
        accuracy_t.append(temp_accuracy_t.item()/temp)
        accuracy_s.append(temp_accuracy_s.item()/temp)
        
        fe.eval()
        dis.eval()
        classifier.eval()
        val_loss = 0
        temp = 0
        y_val_full = np.array([])
        pred_val_full = np.array([])

        for x_val, y_val in val_dataloader:
            x_val = x_val.to(device)
            y_val = y_val.to(device)
            pred_val = classifier(fe(x_val))
            y_val_full = np.concatenate((y_val_full, y_val.detach().cpu().numpy()),axis=0)
            pred_val_full = np.concatenate((pred_val_full, (torch.argmax(pred_val,1)+1).detach().cpu().numpy()),axis=0)
            temp_accuracy_val += ((torch.argmax(pred_val,1)+1)== y_val).to(torch.float).mean().detach().cpu()
            loss = cls_loss(pred_val, y_val-1).detach().cpu()
            val_loss += loss.item() * x_val.size(0)
            temp += 1
        
        f1 = f1_score(y_val_full, pred_val_full, average='weighted', zero_division=0)
        acc = temp_accuracy_val.item()/temp

        val_total_loss = val_loss / len(val_dataloader.dataset)
        val_loss_list.append(val_total_loss)
        print("val_loss :", val_total_loss)

        cm = confusion_matrix(y_val_full, pred_val_full)
        print("\nconfusion_matrix")
        print(cm)
        print()
        print(classification_report(y_val_full, pred_val_full, labels=[1,2,3], zero_division=0))

        accuracy_val.append(acc)
        epochs = epochs + 1
        if val_total_loss > best_loss:
            limit_check += 1
            if(limit_check >= patience and epochs >= nb_epochs/5):
                break
        else:
            best_loss = val_total_loss
            best_val_epoch = epochs
            limit_check = 0
        if  (acc+f1)/2 > best_acc_f1 :
            best_acc_f1 = (acc+f1)/2
            best_acc = acc
            best_f1 = f1
            best_fe_wts = copy.deepcopy(fe.state_dict())
            best_dis_wts = copy.deepcopy(dis.state_dict())
            best_cls_wts = copy.deepcopy(classifier.state_dict())
            best_acc_f1_epoch = epochs 
        
        print(f"best_val_loss : {best_loss}, epoch : {best_val_epoch}")
        print(f"best_acc_score : {best_acc}, epoch : {best_acc_f1_epoch}")
        print(f"best_f1_score : {best_f1}, epoch : {best_acc_f1_epoch}\n")

    print("\naccuracy_t :", sum(accuracy_t)/len(accuracy_t))
    print("accuracy_s :", sum(accuracy_s)/len(accuracy_s))
    print("accuracy_val :", sum(accuracy_val)/len(accuracy_val))
    print(f"best_val_loss : {best_loss}, epoch : {best_val_epoch}")
    print(f"best_acc_score : {best_acc}, epoch : {best_acc_f1_epoch}")
    print(f"best_f1_score : {best_f1}, epoch : {best_acc_f1_epoch}")

    try :
        plt.figure()
        plt.title('val_loss')
        plt.plot(np.arange(1, epochs+1, 1), val_loss_list, label='val')
        plt.xlabel('Epoch')
        plt.ylabel('loss')
        plt.savefig(output_dir+f'/loss/Val_Loss_{label}_{fold}fold.png')
        plt.close()
    except Exception as e :
        print(e)
    finally :
        pass
    
    del x_source
    del x_target
    del x_val
    del y_source
    del y_target
    del y_val
    
    fe.load_state_dict(best_fe_wts)
    dis.load_state_dict(best_dis_wts)
    classifier.load_state_dict(best_cls_wts)

    return fe, dis, classifier

def test_model(fe, classifier, dataloader, label, output_dir, fold=0) :
    true_y = []
    pred_y = []
    cls_y = []
    for x, y in dataloader :
        pred_y.append(torch.argmax(classifier(fe(x.cuda())),1).detach().cpu()+1)
        cls_y.append(classifier(fe(x.cuda())).detach().cpu())
        true_y.append(y)
    true_y = torch.cat(true_y, dim=0).numpy()
    pred_y = torch.cat(pred_y, dim=0).numpy()
    cls_y = torch.cat(cls_y, dim=0).numpy()
    cm = confusion_matrix(true_y, pred_y)
    
    try :
        plt.figure()
        dis_cm = ConfusionMatrixDisplay(confusion_matrix=cm)
        dis_cm.plot(cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.savefig(output_dir+f'/cm/Test_Confusion_Matrix_{label}_{fold}fold.png')
        plt.close()
    except Exception as e :
        print(e)
    finally :
        pass
    
    print()
    print("-"*50)
    print("TEST RESULT")
    accuracy = accuracy_score(true_y, pred_y)
    precision = precision_score(true_y, pred_y, average='weighted', zero_division=0)
    recall = recall_score(true_y, pred_y, average='weighted', zero_division=0)
    f1 = f1_score(true_y, pred_y, average='weighted', zero_division=0)
    roauc = roc_auc_score(true_y, cls_y, average='weighted', multi_class='ovo')

    print(f"accuracy : {accuracy}\nprecision : {precision}\nrecall : {recall}\nf1_score : {f1}\nroc_auc : {roauc}\n")
    print(cm, '\n')
    print(classification_report(true_y, pred_y, labels=[1,2,3], zero_division=0))

    del x
    del y
    
    return accuracy, precision, recall, f1, roauc