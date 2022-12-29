import argparse
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve, auc
from tqdm import tqdm
from model import load_model, FeatureExtractor
import config as c
from utils import *
import matplotlib.pyplot as plt
import torch.nn.functional as F
import PIL
from os.path import join
import os
from copy import deepcopy

def evaluate(model, test_loader, mode='std'):
    times = []
    test_z = []
    test_labels = list()
    with torch.no_grad():
        
        fe = FeatureExtractor()
        for param in fe.parameters():
            param.requires_grad = False
            
        model.to(c.device)
        fe.to(c.device)
        model.eval()
        fe.eval();

        for i, data in enumerate(tqdm(test_loader)):
            inputs, labels, _ = data
            inputs = inputs.cuda()
            start = time.time()
            if not c.pre_extracted:
                inputs = fe(inputs)
            z, jac = nf_forward(model, inputs)
            loss = get_loss(z, jac)
            times.append(time.time()-start)

            z_concat = t2np(concat_maps(z))
            if mode == 'std':
                score = np.std(z_concat ** 2, axis=(1, 2))
            else:
                score = np.mean(z_concat ** 2, axis=(1, 2))
                
            test_z.append(score)
            test_labels.append(t2np(labels))

        print("Inf TIme: {}".format(np.mean(np.array(times))))

        test_labels = np.concatenate(test_labels)
        is_anomaly = np.array([0 if l == 0 else 1 for l in test_labels])

        anomaly_score = np.concatenate(test_z, axis=0)

        print(anomaly_score.shape, is_anomaly.shape)

        fpr, tpr, thresholds = metrics.roc_curve(is_anomaly, anomaly_score)

        roc = roc_auc_score(is_anomaly, anomaly_score)

        lst = []
        for i in range(len(tpr)):
            lst.append(g_means(tpr[i],fpr[i]))
        optimal_idx = np.argmax(np.array(lst))
        optimal_thresh = thresholds[optimal_idx]

    print("ROC: {}".format(roc))

    plt.plot(fpr,tpr)
    idx = np.where(tpr==1)[0][0]
    plt.scatter(fpr[idx],tpr[idx],s=50)
    idx2 = optimal_idx
    plt.scatter(fpr[idx2],tpr[idx2],s=50)
    plt.title('EffB5')

    print("Image AUC: {} FPR: {}".format(roc, fpr[idx]))
    plt.savefig('save_auroc.png')


#Parser
parser = argparse.ArgumentParser(description='Evaluate model')
parser.add_argument('--modelPath', default='/home/edshkim98/synapse/one_class/cs_flow2/cs_flow/cs_flow.pth',
                    help='path for pretrained model')
parser.add_argument('--anomalyMode', default='mean',
                    help='Anomaly scoring method (mean or std)')

args = parser.parse_args()

#Load model
model = get_cs_flow_model()
pretrained = torch.load(args.modelPath)
model.load_state_dict(pretrained)

#Load dataset
trainset = SynapseData(c, c.dataset_path+'/'+c.class_name+'/train', train=True)
testset = SynapseData(c, c.dataset_path+'/'+c.class_name+'/test', train=False, ret_path=True)
train_loader, test_loader = make_dataloaders(trainset, testset)

#Evaluate
evaluate(model, test_loader, mode = args.anomalyMode)
