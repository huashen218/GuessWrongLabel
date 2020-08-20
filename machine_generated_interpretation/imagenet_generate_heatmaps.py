import os
import cv2
import json
import time
import torch
import argparse
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from torchvision.models import vgg16, vgg19
from torchvision import models
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

from imagenet_interpreters import generate_gradsaliency, generate_gradcam, generate_extremal_perturbation, _norm

device = 'cuda' if torch.cuda.is_available() else 'cpu'

np.random.seed(seed=1234)


def save_maps(img, pertb1, pertb2, pertb3, gradcam, saliency, file_name):

    _font = {'family': 'serif',
        'color':  'black',
        'weight': 'medium',
        'size': 6,
        }
    
    plt.figure(figsize=(60,10))
    # plt.figure(figsize=(48,8))
    plt.clf()
    ncols = 6
    
    fig, axs = plt.subplots(1, ncols, constrained_layout=True)
    axs[0].imshow(img)
    axs[0].set_title('original image', fontdict=_font)
    axs[0].axis('off')

    axs[1].imshow(pertb1)
    axs[1].set_title('input_20%', fontdict=_font)
    axs[1].axis('off')

    axs[2].imshow(pertb2)
    axs[2].set_title('input_40%', fontdict=_font)
    axs[2].axis('off')

    axs[3].imshow(pertb3)
    axs[3].set_title('input_50%', fontdict=_font)
    axs[3].axis('off')

    axs[4].imshow(gradcam)
    axs[4].set_title('intermediate', fontdict=_font)
    axs[4].axis('off')

    axs[5].imshow(saliency)
    axs[5].set_title('output', fontdict=_font)
    axs[5].axis('off')

    cb_ax = fig.add_axes([1.00, 0.39, 0.01, 0.22])   # 1: horizontal; 2: vertical; 
    cMap = plt.get_cmap('RdYlBu')
    cbar = fig.colorbar(cm.ScalarMappable(cmap=cMap), cax=cb_ax)
    cbar.outline.set_linewidth(0.1)

    cbar.ax.get_yaxis().set_ticks([])
    for j, lab in enumerate(['0.2','0.4','0.6','0.8']):
        cbar.ax.text(1.0, (2 * j + 1) / 8.0, lab, ha='left', va='center', size=5, family = 'serif')

    fig.savefig(file_name + '.png', bbox_inches='tight', dpi=300)




def _load_incorrect_data(incorrect_img_dir):

    incorrect_data = np.load(incorrect_img_dir)
    incorrect_images = incorrect_data['incorrect_images']
    incorrect_targets = incorrect_data['incorrect_targets']
    incorrect_preds = incorrect_data['incorrect_preds']

    X_incorrect_all = torch.from_numpy(incorrect_images)
    Y_incorrect_all = torch.from_numpy(incorrect_targets)
    Pred_incorrect_all = torch.from_numpy(incorrect_preds)

    return X_incorrect_all, Y_incorrect_all, Pred_incorrect_all



def main(args):

    save_dir = args.save_dir 
    os.makedirs(save_dir) if not os.path.exists(save_dir) else None


    ############################
    # load model
    ############################
    model = models.resnet50(pretrained=True).to(device)


    ############################
    # load labels
    ############################
    with open(args.labels_dir) as f:
        imagenet_labels = json.load(f)


    ############################
    # load inccorect data
    ############################
    X_incorrect_all, Y_incorrect_all, Pred_incorrect_all = _load_incorrect_data(args.incorrect_img_dir)



    ############################
    # random_samples
    ############################
    rand_idx = np.random.randint(X_incorrect_all.size(0), size=args.random_size)

    np.savez(os.path.join(save_dir, f'rand_idx_{len(rand_idx)}.npz'), 
            rand_idx = rand_idx,
            true_y = Y_incorrect_all[rand_idx].detach().cpu().numpy(),
            pred_y = Pred_incorrect_all[rand_idx].detach().cpu().numpy(),
            labels = np.array(imagenet_labels)
    )



    ############################
    # interpret random_samples
    ############################
    for t in tqdm(rand_idx[0:200]):

        start = time.time()

        bx, by, py = X_incorrect_all[t:t+1].to(device), Y_incorrect_all[t:t+1].item(), Pred_incorrect_all[t:t+1].item()
        true_label, pred_label = imagenet_labels[by], imagenet_labels[py]
        img = _norm(bx[0]).detach().cpu().numpy().transpose(1,2,0)

        pertb_pred_mask_005 =  generate_extremal_perturbation(model, bx, py, area=0.2)
        pertb_pred_mask_010 =  generate_extremal_perturbation(model, bx, py, area=0.35)
        pertb_pred_mask_015 =  generate_extremal_perturbation(model, bx, py, area=0.45)
        gradcam_pred_mask = generate_gradcam(model, bx, py)
        saliency_pred_mask = generate_gradsaliency(model, bx, py)

        file_dir = os.path.join(save_dir, f'Idx_{t}')
        save_maps(img, pertb_pred_mask_005, pertb_pred_mask_010, pertb_pred_mask_015, gradcam_pred_mask, saliency_pred_mask, file_dir)

        end = time.time()
        print(f" ====== each time: {end-start} ======")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate ImageNet Saliency Maps')
    parser.add_argument('--incorrect_img_dir', metavar='DIR')
    parser.add_argument('-r', '--random_size', default=500, type=int)
    parser.add_argument('--labels_dir', metavar='DIR', default="/home/hqs5468/hua/workspace/projects/datasets/imagenet/imagenet_labels.json")
    parser.add_argument('--save_dir', dest='save_dir', default='../results')
    parser.add_argument('-d', '--dataset', dest='dataset', default='imagenet')
    parser.add_argument('-m', '--model_name', dest='model_name', default='resnet50')
    parser.add_argument('-b', '--batch-size', default=128, type=int, metavar='N', help='mini-batch size (default: 256)')
    args = parser.parse_args()
    print(args.__dict__)
    main(args)

