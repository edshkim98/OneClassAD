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


localize = True
upscale_mode = 'bilinear'
score_export_dir = join('./viz/scores/', c.modelname)
os.makedirs(score_export_dir, exist_ok=True)
map_export_dir = join('./viz/maps/', c.modelname)
os.makedirs(map_export_dir, exist_ok=True)


def compare_histogram(scores, classes, thresh=2.5, n_bins=64):
    classes = deepcopy(classes)
    classes[classes > 0] = 1
    scores[scores > thresh] = thresh
    bins = np.linspace(np.min(scores), np.max(scores), n_bins)
    scores_norm = scores[classes == 0]
    scores_ano = scores[classes == 1]

    plt.clf()
    plt.hist(scores_norm, bins, alpha=0.5, density=True, label='non-defects', color='cyan', edgecolor="black")
    plt.hist(scores_ano, bins, alpha=0.5, density=True, label='defects', color='crimson', edgecolor="black")

    ticks = np.linspace(0.5, thresh, 5)
    labels = [str(i) for i in ticks[:-1]] + ['>' + str(thresh)]
    plt.xticks(ticks, labels=labels)
    plt.xlabel(r'$-log(p(z))$')
    plt.ylabel('Count (normalized)')
    plt.legend()
    plt.grid(axis='y')
    plt.savefig(join(score_export_dir, 'score_histogram.png'), bbox_inches='tight', pad_inches=0)


def viz_roc(values, classes, class_names):
    def export_roc(values, classes, export_name='all'):
        # Compute ROC curve and ROC area for each class
        classes = deepcopy(classes)
        classes[classes > 0] = 1
        fpr, tpr, _ = roc_curve(classes, values)
        roc_auc = auc(fpr, tpr)

        plt.clf()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='ROC curve (area = %0.4f)' % roc_auc)

        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic for class ' + c.class_name)
        plt.legend(loc="lower right")
        plt.axis('equal')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.savefig(join(score_export_dir, export_name + '.png'))

    export_roc(values, classes)
    for cl in range(1, classes.max() + 1):
        filtered_indices = np.concatenate([np.where(classes == 0)[0], np.where(classes == cl)[0]])
        classes_filtered = classes[filtered_indices]
        values_filtered = values[filtered_indices]
        export_roc(values_filtered, classes_filtered, export_name=class_names[filtered_indices[-1]])


def viz_maps(maps, name, label):
    img_path = img_paths[c.viz_sample_count]
    image = PIL.Image.open(img_path).convert('RGB')
    image = np.array(image)

    map_to_viz = t2np(F.interpolate(maps[0][None, None], size=image.shape[:2], mode=upscale_mode, align_corners=False))[
        0, 0]

    plt.clf()
    plt.imshow(map_to_viz)
    plt.axis('off')
    plt.savefig(join(map_export_dir, name + '_map.jpg'), bbox_inches='tight', pad_inches=0)

    if label > 0:
        plt.clf()
        plt.imshow(image)
        plt.axis('off')
        plt.savefig(join(map_export_dir, name + '_orig.jpg'), bbox_inches='tight', pad_inches=0)
        plt.imshow(map_to_viz, cmap='viridis', alpha=0.3)
        plt.savefig(join(map_export_dir, name + '_overlay.jpg'), bbox_inches='tight', pad_inches=0)
    return


def viz_map_array(maps, labels, n_col=8, subsample=4, max_figures=-1):
    plt.clf()
    fig, subplots = plt.subplots(3, n_col)

    fig_count = -1
    col_count = -1
    for i in range(len(maps)):
        if i % subsample != 0:
            continue

        if labels[i] == 0:
            continue

        col_count = (col_count + 1) % n_col
        if col_count == 0:
            if fig_count >= 0:
                plt.savefig(join(map_export_dir, str(fig_count) + '.jpg'), bbox_inches='tight', pad_inches=0)
                plt.close()
            fig, subplots = plt.subplots(3, n_col, figsize=(22, 8))
            fig_count += 1
            if fig_count == max_figures:
                return

        anomaly_description = img_paths[i].split('/')[-2]
        image = PIL.Image.open(img_paths[i]).convert('RGB')
        image = np.array(image)
        map = t2np(F.interpolate(maps[i][None, None], size=image.shape[:2], mode=upscale_mode, align_corners=False))[
            0, 0]
        subplots[1][col_count].imshow(map)
        subplots[1][col_count].axis('off')
        subplots[0][col_count].imshow(image)
        subplots[0][col_count].axis('off')
        subplots[0][col_count].set_title(c.class_name + ":\n" + anomaly_description)
        subplots[2][col_count].imshow(image)
        subplots[2][col_count].axis('off')
        subplots[2][col_count].imshow(map, cmap='viridis', alpha=0.3)
    for i in range(col_count, n_col):
        subplots[0][i].axis('off')
        subplots[1][i].axis('off')
        subplots[2][i].axis('off')
    if col_count > 0:
        plt.savefig(join(map_export_dir, str(fig_count) + '.jpg'), bbox_inches='tight', pad_inches=0)
    return

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

def evaluate2(model, test_loader):
    model.to(c.device)
    model.eval()
    if not c.pre_extracted:
        fe = FeatureExtractor()
        fe.eval()
        fe.to(c.device)
        for param in fe.parameters():
            param.requires_grad = False

    print('\nCompute maps, loss and scores on test set:')
    anomaly_score = list()
    test_labels = list()
    c.viz_sample_count = 0
    all_maps = list()
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader, disable=c.hide_tqdm_bar)):
            inputs, labels = preprocess_batch(data)
            if not c.pre_extracted:
                inputs = fe(inputs)
            z = model(inputs)

            z_concat = t2np(concat_maps(z))
            nll_score = np.mean(z_concat ** 2 / 2, axis=(1, 2))
            anomaly_score.append(nll_score)
            test_labels.append(t2np(labels))

            if localize:
                z_grouped = list()
                likelihood_grouped = list()
                for i in range(len(z)):
                    z_grouped.append(z[i].view(-1, *z[i].shape[1:]))
                    likelihood_grouped.append(torch.mean(z_grouped[-1] ** 2, dim=(1,)))
                all_maps.extend(likelihood_grouped[0])
                for i_l, l in enumerate(t2np(labels)):
                    # viz_maps([lg[i_l] for lg in likelihood_grouped], c.modelname + '_' + str(c.viz_sample_count), label=l, show_scales = 1)
                    c.viz_sample_count += 1

    anomaly_score = np.concatenate(anomaly_score)
    test_labels = np.concatenate(test_labels)

    compare_histogram(anomaly_score, test_labels)

    class_names = [img_path.split('/')[-2] for img_path in img_paths]
    viz_roc(anomaly_score, test_labels, class_names)

    test_labels = np.array([1 if l > 0 else 0 for l in test_labels])
    auc_score = roc_auc_score(test_labels, anomaly_score)
    print('AUC:', auc_score)

    if localize:
        viz_map_array(all_maps, test_labels)

    return

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

# train_set, test_set = load_datasets(c.dataset_path, c.class_name)
# img_paths = test_set.paths if c.pre_extracted else [p for p, l in test_set.samples]
# _, test_loader = make_dataloaders(train_set, test_set)
# mod = load_model(c.modelname)
