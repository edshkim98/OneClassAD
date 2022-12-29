import numpy as np
import torch
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import config as c
from model import get_cs_flow_model, save_model, FeatureExtractor, nf_forward
from utils import *
import time
from sklearn import metrics
import pandas as pd

def calc_roc(output):
    preds_img = []
    ys = []
    for i in range(len(output)):
        preds_img.append(output[i]["pred"])
        ys.append(output[i]['label'].cpu())
    
    preds_img = np.array(preds_img)
    ys = np.array(ys)
    preds_img = (preds_img-preds_img.min())/(preds_img.max()-preds_img.min())

    fpr, tpr, thresholds = metrics.roc_curve(ys, preds_img, pos_label=1)
    idx = np.where(tpr==1)[0][0]

    roc = metrics.auc(fpr, tpr)
    
    print("Image AUC: {} FPR: {}".format(roc, fpr[idx]))
    return roc, fpr, tpr

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    
def train(train_loader, test_loader):
    best = 1.0
    model = get_cs_flow_model()
    if c.finetune:
        pretrained = torch.load(os.getcwd()+"/models/tmbl/cs_flow_couplingx4_warm_std.pth")
        model.load_state_dict(pretrained)
        print("Finetuning!")
        for param in model.parameters():
            param.requires_grad = True
    optimizer = torch.optim.Adam(model.parameters(), lr=c.lr_init, eps=1e-04, weight_decay=1e-5)
    steps = len(train_loader)*10
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0 = steps, T_mult = 1, eta_min=0.00001)

    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    model.to(c.device)
    if not c.pre_extracted:
        fe = FeatureExtractor(c)
        fe.eval()
        fe.to(c.device)
        for param in fe.parameters():
            param.requires_grad = False
        print("Features not pre-extracted!")
    else:
        print("Features pre-extracted!")

    z_obs = Score_Observer('AUROC')
    avg_valid_losses = []
    for epoch in range(c.meta_epochs):
        # train some epochs
        model.train()
        if c.verbose:
            print(F'\nTrain epoch {epoch}')
        for sub_epoch in range(c.sub_epochs):
            train_loss = list()

            for i, data in enumerate(tqdm(train_loader, disable=c.hide_tqdm_bar)):
                optimizer.zero_grad()
                
                try:
                    inputs, labels, path = preprocess_batch(data)  # move to device and reshapes
                except:
                    inputs, labels = preprocess_batch(data)  # move to device and reshapes
                if not c.pre_extracted:
                    inputs = fe(inputs)

                z, jac = nf_forward(model, inputs)

                loss = get_loss(z, jac)
                train_loss.append(t2np(loss))

                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), c.max_grad_norm)
                optimizer.step()
                scheduler.step()
                
            mean_train_loss = np.mean(train_loss)
            if c.verbose and epoch == 0 and sub_epoch % 4 == 0:
                print('Epoch: {:d}.{:d} \t train loss: {:.4f}'.format(epoch, sub_epoch, mean_train_loss))

            # evaluate
            model.eval()
            if c.verbose:
                print('\nCompute loss and scores on test set:')
            test_loss = list()
            test_z = list()
            test_labels = list()
            times = []
            
            with torch.no_grad():
                for i, data in enumerate(tqdm(test_loader, disable=c.hide_tqdm_bar)):
                    try:
                        inputs, labels, path = preprocess_batch(data)  # move to device and reshapes
                    except:
                        inputs, labels = preprocess_batch(data)  # move to device and reshapes
                    start = time.time()
                    if not c.pre_extracted:
                        inputs = fe(inputs)

                    z, jac = nf_forward(model, inputs)
                    loss = get_loss(z, jac)
                    end = time.time()

                    times.append(end-start)

                    z_concat = t2np(concat_maps(z))
                    score = np.mean(z_concat ** 2, axis=(1, 2))
                    test_z.append(score)
                    test_loss.append(t2np(loss))
                    test_labels.append(t2np(labels))
            print("Inf TIme: {}".format(np.mean(np.array(times))))
            test_loss = np.mean(np.array(test_loss))

            test_labels = np.concatenate(test_labels)
            is_anomaly = np.array([0 if l == 0 else 1 for l in test_labels])

            anomaly_score = np.concatenate(test_z, axis=0)
            
            fpr, tpr, thresholds = metrics.roc_curve(is_anomaly, anomaly_score)
            idx = np.where(tpr==1)[0][0]
            
            roc = roc_auc_score(is_anomaly, anomaly_score)
            
            if c.verbose:
                print('Epoch: {:d} \t test_loss: {:.4f} AUC: {:.4f} FPR: {}'.format(epoch, test_loss, roc, fpr[idx]))
            if best > fpr[idx]:
                best = fpr[idx]
                print("Best model!")
                np.save('fpr.npy', fpr)
                np.save('tpr.npy', tpr)
                torch.save(model.state_dict(), os.getcwd()+"/models/cs_flow.pth")
                print("Model Saved")
#             val_loss_save = pd.DataFrame({'loss': avg_valid_losses}).to_csv(os.path.join('/home/edshkim98/synapse/one_class/cs-flow2/cs-flow/val_loss.csv'), index=False)
            
        z_obs.update(roc_auc_score(is_anomaly, anomaly_score), epoch,
                     print_score=c.verbose or epoch == c.meta_epochs - 1)
        
#     if c.save_model:
#         model.to('cpu')
#         save_model(model, c.modelname)

    return z_obs.max_score, z_obs.last, z_obs.min_loss_score
