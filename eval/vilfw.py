import os.path
import torch
import numpy as np

from ghostfacenetsv2 import GhostFaceNetsV2
from torch.nn import  functional as F
from tqdm import tqdm
import cv2
from torch.utils.data import Dataset
from torchvision import transforms as trans
import glog as log
def get_subset(container, subset_bounds):
    """Returns a subset of the given list with respect to the list of bounds"""
    subset = []
    for bound in subset_bounds:
        subset += container[bound[0]: bound[1]]
    return subset


def get_roc(scores_with_gt, n_threshs=400):
    """Computes a ROC cureve on the LFW dataset"""
    thresholds = np.linspace(0., 4., n_threshs)

    fp_rates = []
    tp_rates = []

    for threshold in thresholds:
        fp = 0
        tp = 0
        for score_with_gt in scores_with_gt:
            predict_same = score_with_gt['score'] < threshold
            actual_same = score_with_gt['is_same']

            if predict_same and actual_same:
                tp += 1
            elif predict_same and not actual_same:
                fp += 1

        fp_rates.append(float(fp) / len(scores_with_gt) * 2)
        tp_rates.append(float(tp) / len(scores_with_gt) * 2)

    return np.array(fp_rates), np.array(tp_rates)

def get_auc(fprs, tprs):
    """Computes AUC under a ROC curve"""
    sorted_fprs, sorted_tprs = zip(*sorted(zip(*(fprs, tprs))))
    sorted_fprs = list(sorted_fprs)
    sorted_tprs = list(sorted_tprs)
    if sorted_fprs[-1] != 1.0:
        sorted_fprs.append(1.0)
        sorted_tprs.append(sorted_tprs[-1])
    return np.trapz(sorted_tprs, sorted_fprs)


def save_roc(fp_rates, tp_rates, fname):
    assert fp_rates.shape[0] == tp_rates.shape[0]
    with open(fname + '.txt', 'w') as f:
        for i in range(fp_rates.shape[0]):
            f.write('{} {}\n'.format(fp_rates[i], tp_rates[i]))
@torch.no_grad()
def compute_embeddings_vilfw( val_loader, model, batch_size,
                           pdist=lambda x, y: 1. - F.cosine_similarity(x, y)):
    """Computes embeddings of all images from the LFW dataset using PyTorch"""


    scores_with_gt = []


    for batch_idx, data in enumerate(tqdm(val_loader, 'Computing embeddings')):
        images_1 = data[0]
        images_2 = data[1]
        is_same = data[2]
        if torch.cuda.is_available() :
            images_1 = images_1.cuda()
            images_2 = images_2.cuda()
        emb_1 = model(images_1)
        emb_2 = model(images_2)
        scores = pdist(emb_1, emb_2).data.cpu().numpy()

        for i, _ in enumerate(scores):
            scores_with_gt.append({'score': scores[i], 'is_same': is_same[i], 'idx': batch_idx*batch_size + i})


    return scores_with_gt

def compute_optimal_thresh(scores_with_gt):
    """Computes an optimal threshold for pairwise face verification"""
    pos_scores = []
    neg_scores = []
    for score_with_gt in scores_with_gt:
        if score_with_gt['is_same']:
            pos_scores.append(score_with_gt['score'])
        else:
            neg_scores.append(score_with_gt['score'])

    hist_pos, bins = np.histogram(np.array(pos_scores), 60)
    hist_neg, _ = np.histogram(np.array(neg_scores), bins)

    intersection_bins = []

    for i in range(1, len(hist_neg)):
        if hist_pos[i - 1] >= hist_neg[i - 1] and 0.05 < hist_pos[i] <= hist_neg[i]:
            intersection_bins.append(bins[i])

    if not intersection_bins:
        intersection_bins.append(0.5)

    return np.mean(intersection_bins)

def evaluate( var_loader, model, val_batch_size=4,  snap_name='', verbose=True, show_failed=False):



    """Computes the LFW score of given model"""
    # if verbose and isinstance(model, torch.nn.Module):
    #     log.info('Face recognition model config:')
    #     log.info(model)

    scores_with_gt = compute_embeddings_vilfw( var_loader, model, val_batch_size)
    num_pairs = len(scores_with_gt)

    subsets = []
    for i in range(10):
        lower_bnd = i * num_pairs // 10
        upper_bnd = (i + 1) * num_pairs // 10
        subset_test = [(lower_bnd, upper_bnd)]
        subset_train = [(0, lower_bnd), (upper_bnd, num_pairs)]
        subsets.append({'test': subset_test, 'train': subset_train})

    same_scores = []
    diff_scores = []
    val_scores = []
    threshs = []
    mean_fpr = np.zeros(400)
    mean_tpr = np.zeros(400)
    failed_pairs = []

    for subset in tqdm(subsets, '{} evaluation'.format(snap_name), disable=not verbose):
        train_list = get_subset(scores_with_gt, subset['train'])
        optimal_thresh = compute_optimal_thresh(train_list)
        threshs.append(optimal_thresh)

        test_list = get_subset(scores_with_gt, subset['test'])
        same_correct = 0
        diff_correct = 0
        pos_pairs_num = neg_pairs_num = len(test_list) // 2

        for score_with_gt in test_list:
            if score_with_gt['score'] < optimal_thresh and score_with_gt['is_same']:
                same_correct += 1
            elif score_with_gt['score'] >= optimal_thresh and not score_with_gt['is_same']:
                diff_correct += 1

            if score_with_gt['score'] >= optimal_thresh and score_with_gt['is_same']:
                failed_pairs.append(score_with_gt['idx'])
            if score_with_gt['score'] < optimal_thresh and not score_with_gt['is_same']:
                failed_pairs.append(score_with_gt['idx'])

        same_scores.append(float(same_correct) / pos_pairs_num)
        diff_scores.append(float(diff_correct) / neg_pairs_num)
        val_scores.append(0.5*(same_scores[-1] + diff_scores[-1]))

        fprs, tprs = get_roc(test_list, mean_fpr.shape[0])
        mean_fpr = mean_fpr + fprs
        mean_tpr = mean_tpr + tprs

    mean_fpr /= 10
    mean_tpr /= 10

    same_acc = np.mean(same_scores)
    diff_acc = np.mean(diff_scores)
    overall_acc = np.mean(val_scores)
    auc = get_auc(mean_fpr, mean_tpr)

    # if show_failed:
    #     log.info('Number of misclassified pairs: {}'.format(len(failed_pairs)))
    #     for pair in failed_pairs:
    #         dataset.show_item(pair)

    if verbose:
        log.info('Accuracy/Val_same_accuracy mean: {0:.4f}'.format(same_acc))
        log.info('Accuracy/Val_diff_accuracy mean: {0:.4f}'.format(diff_acc))
        log.info('Accuracy/Val_accuracy mean: {0:.4f}'.format(overall_acc))
        log.info('Accuracy/Val_accuracy std dev: {0:.4f}'.format(np.std(val_scores)))
        log.info('AUC: {0:.4f}'.format(auc))
        log.info('Estimated threshold: {0:.4f}'.format(np.mean(threshs)))

    return same_acc, diff_acc, overall_acc, auc,threshs


class VILWFDataset(Dataset):
    def __init__(self,root_path,all_img):
        self.root_path = root_path
        self.all_img = all_img
        self.trans =trans.Compose([
            trans.ToTensor(),
            trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    def __getitem__(self,idx):
        record =  self.all_img.iloc[idx]
        img1 = record["img1"]
        img2 = record["img2"]
        label = record["label"]
        path1= os.path.join(self.root_path,img1)
        path2 = os.path.join(self.root_path,img2)
        img1 = cv2.imdecode(np.fromfile(path1, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        img2 = cv2.imdecode(np.fromfile(path2, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        img1 = self.trans(img1)
        img2 = self.trans(img2)
        return img1, img2, label

    def __len__(self):
        return len(self.all_img)


if __name__ == '__main__':


    from mobilefacenet import MobileFaceNet
    from torch.utils.data.dataloader import DataLoader
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = MobileFaceNet(512).to(device)
    model.load_state_dict(
        torch.load('../Weights/MobileFace_Net', map_location=lambda storage, loc: storage))
    model.eval()
    import  pandas as pd
    all_img = pd.read_csv("output.csv")
    data = VILWFDataset("..\VILFWCut",all_img)
    dataloader = DataLoader(data,4)
    same_acc, diff_acc, overall_acc, auc,threshs=evaluate(dataloader,model,4)
    print(same_acc,diff_acc,overall_acc,auc)
