""" script for computing the fid of a trained model when compared with the dataset images """
import argparse
import tempfile
from pathlib import Path
import torch.nn.functional as F
import imageio as imageio
import torch
torch.cuda.empty_cache()
#from cleanfid import fid
from torch.backends import cudnn
import math
from tqdm import tqdm
import os
import flopy.utils.binaryfile as bf
from networks import create_generator_from_saved_model
#from utils import post_process_generated_images
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from scipy import array, linalg, dot



# turn fast mode on
cudnn.benchmark = True


nr = 100
maxIter = 6  # iteration times
alpha = np.zeros(maxIter)
for pp in range(maxIter):
    alpha[pp] = (2**(maxIter-pp))

def generate_condition(input_matrix):
    # ref_k_array = np.loadtxt("k_array_ref_gan.txt")
    ref_k_array = torch.as_tensor(input_matrix, dtype=torch.float32)
    well_points = np.zeros(ref_k_array.shape, dtype=int)

    xs = [4, 4, 4, 64, 64, 64, 124, 124, 124]
    ys = [4, 64, 124, 4, 64, 124, 4, 64, 124]
    #xs = [4, 4, 4, 4, 4, 34, 34, 34, 34, 34, 64, 64, 64, 64, 64, 94, 94, 94, 94, 94, 124, 124, 124, 124, 124]
    #ys = [4, 34, 64, 94, 124, 4, 34, 64, 94, 124, 4, 34, 64, 94, 124, 4, 34, 64, 94, 124, 4, 34, 64, 94, 124]
    #xs = [4, 4, 4, 4, 4, 4, 4, 24, 24, 24, 24, 24, 24, 24, 44, 44, 44, 44, 44, 44, 44, 64, 64, 64, 64, 64, 64, 64, 84,
          #4, 84, 84, 84, 84, 84, 104, 104, 104, 104, 104, 104, 104, 124, 124, 124, 124, 124, 124, 124]
    #ys = [4, 24, 44, 64, 84, 104, 124, 4, 24, 44, 64, 84, 104, 124, 4, 24, 44, 64, 84, 104, 124, 4, 24, 44, 64, 84, 104, 124,
          #4, 24, 44, 64, 84, 104, 124, 4, 24, 44, 64, 84, 104, 124, 4, 24, 44, 64, 84, 104, 124]
    #xs = [4, 4, 4, 4, 4, 4, 4, 4, 4, 19, 19, 19, 19, 19, 19, 19, 19, 19, 34, 34, 34, 34, 34, 34, 34, 34, 34, 49, 49, 49,
         #49, 49, 49, 49, 49, 49, 64, 64, 64, 64, 64, 64, 64, 64, 64, 79, 79, 79, 79, 79, 79, 79, 79, 79, 94, 94, 94, 94,
          #94, 94, 94, 94, 94, 109, 109, 109, 109, 109, 109, 109, 109, 109, 124, 124, 124, 124, 124, 124, 124, 124, 124]
    #ys = [4, 19, 34, 49, 65, 79, 94, 109, 124, 4, 19, 34, 49, 65, 79, 94, 109, 124, 4, 19, 34, 49, 65, 79, 94, 109, 124,
          #4, 19, 34, 49, 65, 79, 94, 109, 124, 4, 19, 34, 49, 65, 79, 94, 109, 124, 4, 19, 34, 49, 65, 79, 94, 109, 124,
          #4, 19, 34, 49, 65, 79, 94, 109, 124, 4, 19, 34, 49, 65, 79, 94, 109, 124, 4, 19, 34, 49, 65, 79, 94, 109, 124,]
    well_points[:, :, xs, ys] = 1
    random_matrix = torch.from_numpy(well_points).type('torch.FloatTensor')

    # Enlarge condition points
    sf = 2
    avg_downsampler = torch.nn.MaxPool2d((sf, sf), stride=(sf, sf))
    random_matrix = avg_downsampler(random_matrix)
    random_matrix = F.interpolate(random_matrix, scale_factor=sf, mode="nearest")

    output_matrix = ref_k_array * random_matrix    #.cuda()   .cuda()
    # output_matrix = torch.zeros_like(input_matrix)
    return output_matrix, torch.as_tensor(random_matrix, dtype=torch.float32)   #.cuda()      # .cuda()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("ProGAN fid_score computation tool",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter, )

    # fmt: off
    # required arguments
    parser.add_argument("--model_path", action="store", type=Path,
                        default=r'D:\pro_gan_pytorch-master_1\pro_gan_pytorch-master\pro_gan_pytorch-master\pro_gan_pytorch\mod_img\models\depth_7_epoch_30.bin',
                        help="path to the trained_model.bin file")
    parser.add_argument("--dataset_path", action="store", type=Path,
                        default=r'C:\Users\101100464\Downloads\Progressive-GAN-pytorch-trimmed\Progressive-GAN-pytorch-trimmed\train_im\ti_image',
                        help="path to the directory containing the images from the dataset. "
                             "Note that this needs to be a flat directory")

    parser.add_argument("--batch_size", action="store", type=int, default= 100, required=False,
                        help="batch size used for generating random images")
    parser.add_argument("--num_generated_images", action="store", type=int, default=500, required=False,
                        help="number of generated images used for computing the FID")
    # fmt: on

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = create_generator_from_saved_model(args.model_path)  #.to(device)




    #hs = np.zeros((h_n_ob * len(Time_H), 2), dtype=np.float32)
    #hss = np.zeros((h_n_ob * len(Time_H), nr), dtype=np.float32)

    #for i in range(1, h_n_ob + 1):
        #locals()['h_idx_' + str(i)] = (layer, eval('Head_' + str(i) + '_x'), eval('Head_' + str(i) + '_y'))

    random_latents = torch.randn(args.batch_size, 128)  # , device=device



    ln_K_ref = np.loadtxt(r'ref')  # the path of reference K
    plt.imshow(ln_K_ref)
    plt.axis('off')
    #plt.colorbar()

    #for i in range(1, h_n_ob + 1):  # the location of wells
        #plt.scatter(eval('Head_' + str(i) + '_x'), eval('Head_' + str(i) + '_y'), marker='o', c='r', edgecolors='r',
                    #s=20)
    plt.savefig('Ref_K.png')
    plt.close()



    tss = np.ones((100, nr+1 ), dtype=np.float32)
    idx = (0, int(128 / 2) - 1, int(128 / 2) - 1)

    #ln_K_ref = np.log(ln_K_ref)
    #max_ln = np.max(ln_K_ref)
    #ln_K_ref = ln_K_ref/max_ln
    #ln_K_ref = np.where(ln_K_ref<0, 0, 1)
    convert_tensor = torch.from_numpy(ln_K_ref)
    convert_tensor = convert_tensor.reshape(1, 1, 128, 128)
    cond_array, cond_mask = generate_condition(convert_tensor)
    cond_plot = cond_array.reshape(128, 128)
    #cond_plot = cond_plot.cpu().detach().numpy()
    #plt.imshow(cond_plot)
    #plt.savefig('cond_81')
    cond_array = cond_array.repeat(100, 1, 1, 1)
    cond_np = cond_array.cpu().detach().numpy()

    cmap_well = plt.cm.get_cmap("viridis").copy()
    # cmap_well.copy()
    cmap_well.set_bad(color='white')

    for pp in range(0, maxIter):
        k_array = generator(random_latents, cond_array).to(device)
        k_array = k_array.cpu().detach().numpy()
        k_array = k_array.reshape(100, 128, 128)
        Average_K = np.array(k_array.mean(axis=0))  # the average of K
        Variance_K = np.array(np.var(k_array, axis=0))  # the variance of K
        np.savetxt('Average_K_' + str(pp) + '.txt', Average_K)
        np.savetxt('Variance_K_' + str(pp) + '.txt', Variance_K)
        plt.imshow(Average_K)
        plt.colorbar()
        plt.savefig('Average_K_' + str(pp) + '.png')
        plt.close()

        plt.imshow(Variance_K)
        plt.colorbar()
        plt.savefig('Variance_K_' + str(pp) + '.png')
        plt.close()

        plt.figure(figsize=(10, 8))  # Realizations of K
        plt.suptitle('First 9 K_' + str(pp))
        plt.subplots_adjust(wspace=0.5, hspace=1)
        for r in range(0, 9):
            np.savetxt('First 9 K_Iter ' + str(pp) + '_' + str(r + 1) + '.txt', k_array[r, :, :])
            plt.subplot(3, 3, r + 1)
            plt.imshow(k_array[r, :, :])
            plt.colorbar()
        plt.savefig('First 9 K_' + str(pp) + '.png')
        plt.close()

