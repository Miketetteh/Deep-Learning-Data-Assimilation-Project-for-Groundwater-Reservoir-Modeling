""" script for computing the fid of a trained model when compared with the dataset images """
import argparse
import tempfile
from pathlib import Path
import torch.nn.functional as F
import imageio as imageio
import torch
torch.cuda.empty_cache()
from cleanfid import fid
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



# turn fast mode on
cudnn.benchmark = True

modelname = 'test'
nr = 100
maxIter = 6   # iteration times
alpha = np.zeros(maxIter)
for pp in range(maxIter):
    alpha[pp] = (2**(maxIter-pp))

def generate_condition(input_matrix):
    # ref_k_array = np.loadtxt("k_array_ref_gan.txt")
    ref_k_array = torch.as_tensor(input_matrix, dtype=torch.float32)
    well_points = np.zeros(ref_k_array.shape, dtype=int)

    #xs = [4, 4, 4, 64, 64, 64, 124, 124, 124]
    #ys = [4, 64, 124, 4, 64, 124, 4, 64, 124]
    xs = [4, 4, 4, 4, 4, 34, 34, 34, 34, 34, 64, 64, 64, 64, 64, 94, 94, 94, 94, 94, 124, 124, 124, 124, 124]
    ys = [4, 34, 64, 94, 124, 4, 34, 64, 94, 124, 4, 34, 64, 94, 124, 4, 34, 64, 94, 124, 4, 34, 64, 94, 124]
    well_points[:, :, xs, ys] = 1
    random_matrix = torch.from_numpy(well_points).type('torch.FloatTensor')

    # Enlarge condition points
    sf = 1
    avg_downsampler = torch.nn.MaxPool2d((sf, sf), stride=(sf, sf))
    random_matrix = avg_downsampler(random_matrix)
    random_matrix = F.interpolate(random_matrix, scale_factor=sf, mode="nearest")

    output_matrix = ref_k_array * random_matrix        #.cuda()   .cuda()
    # output_matrix = torch.zeros_like(input_matrix)
    return output_matrix, torch.as_tensor(random_matrix, dtype=torch.float32)   #.cuda()   .cuda()


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

    parser.add_argument("--batch_size", action="store", type=int, default=100, required=False,
                        help="batch size used for generating random images")
    parser.add_argument("--num_generated_images", action="store", type=int, default=500, required=False,
                        help="number of generated images used for computing the FID")
    # fmt: on

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = create_generator_from_saved_model(args.model_path)#.to(device)

    h_n_ob = 25  # the total number of head observation points
    h_spacing = 30  # the spacing between two wells
    h_Start = 4
    h_End = 124
    Head_map = int(h_n_ob ** 0.5)

    h_k = 1
    layer = 0
    for i in range(h_Start, h_End + h_spacing, h_spacing):
        for j in range(h_Start, h_End + h_spacing, h_spacing):
            locals()['Head_' + str(h_k) + '_x'] = j
            locals()['Head_' + str(h_k) + '_y'] = i
            h_k = h_k + 1

    Ref_H = np.loadtxt('Ref_Head.txt')  # reference head
    Time_H = np.loadtxt('Ref_Time_Head.txt')  # simulation time period
    Ref_Head = Ref_H.reshape(h_n_ob, len(Time_H))
    Ref_Head = Ref_Head.T

    hs = np.zeros((h_n_ob * len(Time_H), 2), dtype=np.float32)
    hss = np.zeros((h_n_ob * len(Time_H), nr), dtype=np.float32)

    for i in range(1, h_n_ob + 1):
        locals()['h_idx_' + str(i)] = (layer, eval('Head_' + str(i) + '_x'), eval('Head_' + str(i) + '_y'))

    random_latents = torch.randn(args.batch_size, 128)  #, device=device
    #random_latents1 = torch.randn(50, 128, device=device)

    Ref_H = Ref_H.reshape(Ref_H.shape[0], 1)
    Ref = Ref_H
    CD = np.eye(len(Ref))
    mu = 0
    sigma = 0.5
    np.random.seed(0)
    s = np.random.normal(mu, sigma, len(Ref))
    s = s.reshape(s.shape[0], 1)
    d_obs = Ref + s
    Sigma = (d_obs - Ref) ** 2
    for i in range(0, len(Ref)):
        CD[i, i] = Sigma[i]

    ln_K_ref = np.loadtxt(r'ref')  # the path of reference K
    plt.imshow(ln_K_ref)
    plt.colorbar()

    for i in range(1, h_n_ob + 1):  # the location of wells
        plt.scatter(eval('Head_' + str(i) + '_x'), eval('Head_' + str(i) + '_y'), marker='o', c='r', edgecolors='r',
                    s=20)
    plt.savefig('Ref_K.png')
    plt.close()

    #ln_K_ref = np.log(ln_K_ref)
    #max_ln = np.max(ln_K_ref)
    #ln_K_ref = ln_K_ref/max_ln
    #ln_K_ref = np.where(ln_K_ref<0, 0, 1)
    convert_tensor = torch.from_numpy(ln_K_ref)
    convert_tensor = convert_tensor.reshape(1, 1, 128, 128)
    cond_array, cond_mask = generate_condition(convert_tensor)
    cond_array = cond_array.repeat(100, 1, 1, 1)
    #cond_array = cond_array.repeat(25, 1, 1, 1)
    #cond_array1 = cond_array.repeat(2, 1, 1, 1)
    # cond_np = cond_array.cpu().detach().numpy()

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

        for i in range(0, nr):
            print(i)
            np.savetxt('K.txt', np.power(10, k_array[i, :, :]))                        #  np.exp(k_array[i, :, :]))
            f1 = open('test.lpf', 'r+')
            infos_1 = f1.readlines()
            f1.close()
            f2 = open('K.txt', 'r+')
            infos_2 = f2.readlines()
            f2.close()
            infos_1[7:(len(infos_1) - 3)] = infos_2
            infos_3 = infos_1
            f3 = open('test.lpf', 'w+')
            f3.writelines(infos_3)
            f3.close()

            os.system('mf2k.exe test.mfn')  # name file for MODFLOW
            headobj = bf.HeadFile(modelname + '.hed')  # Name of the head file
            for t in range(1, h_n_ob + 1):
                locals()['hs_' + str(t)] = headobj.get_ts(eval('h_idx_' + str(t)))
                hs[(t - 1) * len(Time_H):t * len(Time_H), :] = eval('hs_' + str(t))
            hss[:, i] = hs[:, 1]

        yf = random_latents.cpu().detach().numpy()
        yf = yf.transpose()
        ym = np.array(yf.mean(axis=1))  # Mean of the y_f
        ym = ym.reshape(ym.shape[0], 1)
        dmf = yf - ym

        df = hss
        dm = np.array(df.mean(axis=1))
        dm = dm.reshape(dm.shape[0], 1)
        ddf = df - dm

        Cmd_f = (np.dot(dmf, ddf.T)) / (nr - 1);  # The cros-covariance matrix
        Cdd_f = (np.dot(ddf, ddf.T)) / (nr - 1);  # The auto covariance of predicted data

        R = np.linalg.cholesky(CD)  # Matriz triangular inferior
        U = R.T  # Matriz R transposta
        p, w = np.linalg.eig(CD)

        d_obs = d_obs.reshape(d_obs.shape[0], 1)
        aux = np.repeat(d_obs, nr, axis=1)
        mean = 0 * (d_obs.T)
        noise = np.random.multivariate_normal(mean[0], np.eye(len(Ref)), nr).T

        d_uc = aux + math.sqrt(alpha[pp]) * np.dot(U, noise)

        # Analysis step
        varn = 1 - 1 / math.pow(10, 2)

        u, s, vh = np.linalg.svd(Cdd_f + alpha[pp] * CD);
        v = vh.T
        diagonal = s
        for i in range(len(diagonal)):
            if (sum(diagonal[0:i + 1])) / (sum(diagonal)) > varn:
                diagonal = diagonal[0:i + 1]
                break

        u = u[:, 0:i + 1]
        v = v[:, 0:i + 1]
        ess = np.diag(diagonal ** (-1))
        K = np.dot(Cmd_f, (np.dot(np.dot(v, ess), (u.T))))

        ya = yf + (np.dot(K, (d_uc - df)))
        ya = ya.transpose()
        random_latents = torch.from_numpy(ya).detach().float()

        Head_x = hss.reshape(h_n_ob, len(Time_H), nr)
        Head = np.ones((len(Time_H), h_n_ob, nr), dtype=np.float32)
        for i in range(0, nr):
            Head[:, :, i] = Head_x[:, :, i].T
        plt.figure(figsize=(10, 8))
        plt.suptitle('Figure_' + str(pp))
        plt.subplots_adjust(wspace=0.5, hspace=1)
        for j in range(0, h_n_ob):
            np.savetxt('Iter ' + str(pp) + '_Head_' + str(j + 1) + '.txt', Head[:, j, :])
            plt.subplot(Head_map, Head_map, j + 1)
            plt.title('well_' + str(j + 1))
            plt.xlabel('time')
            plt.ylabel('head')
            for k in range(0, nr):
                plt.plot(Time_H, Head[:, j, k], 'b')
                plt.plot(Time_H, Ref_Head[:, j], 'r-')
        plt.savefig('Head_' + str(pp) + '.png')
        plt.close()

    # final head

    hs = np.zeros((h_n_ob * len(Time_H), 2), dtype=np.float32)
    hss = np.zeros((h_n_ob * len(Time_H), nr), dtype=np.float32)
    k_array = generator(random_latents, cond_array).to(device)
    k_array = k_array.cpu().detach().numpy()
    k_array = k_array.reshape(100, 128, 128)
    Average_K = np.array(k_array.mean(axis=0))  # the average of K
    Variance_K = np.array(np.var(k_array, axis=0))  # the variance of K
    np.savetxt('Average_K_' + str(maxIter) + '.txt', Average_K)
    np.savetxt('Variance_K_' + str(maxIter) + '.txt', Variance_K)
    plt.imshow(Average_K)
    plt.colorbar()
    plt.savefig('Average_K_' + str(maxIter) + '.png')
    plt.close()

    plt.imshow(Variance_K)
    plt.colorbar()
    plt.savefig('Variance_K_' + str(maxIter) + '.png')
    plt.close()

    plt.figure(figsize=(10, 8))  # Realizations of K
    plt.suptitle('First 9 K_' + str(maxIter))
    plt.subplots_adjust(wspace=0.5, hspace=1)
    for r in range(0, 9):
        np.savetxt('First 9 K_Iter ' + str(maxIter) + '_' + str(r + 1) + '.txt', k_array[r, :, :])
        plt.subplot(3, 3, r + 1)
        plt.imshow(k_array[r, :, :])
        plt.colorbar()
    plt.savefig('First 9 K_' + str(maxIter) + '.png')
    plt.close()

    for i in range(0, nr):
        print(i)
        np.savetxt('K.txt', np.power(10, k_array[i, :, :]))                                                  #)np.exp(k_array[i, :, :]))
        f1 = open('test.lpf', 'r+')
        infos_1 = f1.readlines()
        f1.close()
        f2 = open('K.txt', 'r+')
        infos_2 = f2.readlines()
        f2.close()
        infos_1[7:(len(infos_1) - 3)] = infos_2
        infos_3 = infos_1
        f3 = open('test.lpf', 'w+')
        f3.writelines(infos_3)
        f3.close()

        os.system('mf2k.exe test.mfn')  # modelname.mfn
        headobj = bf.HeadFile(modelname + '.hed')  # Name of the head file
        for t in range(1, h_n_ob + 1):
            locals()['hs_' + str(t)] = headobj.get_ts(eval('h_idx_' + str(t)))
            hs[(t - 1) * len(Time_H):(t * len(Time_H)), :] = eval('hs_' + str(t))
        hss[:, i] = hs[:, 1]

    Head_x = hss.reshape(h_n_ob, len(Time_H), nr)
    Head = np.ones((len(Time_H), h_n_ob, nr), dtype=np.float32)
    for i in range(0, nr):
        Head[:, :, i] = Head_x[:, :, i].T
    plt.figure(figsize=(10, 8))
    plt.suptitle('Figure_' + str(maxIter))
    plt.subplots_adjust(wspace=0.5, hspace=1)
    for j in range(0, h_n_ob):
        np.savetxt('Iter ' + str(maxIter) + '_Head_' + str(j + 1) + '.txt', Head[:, j, :])
        plt.subplot(Head_map, Head_map, j + 1)
        plt.title('well_' + str(j + 1))
        plt.xlabel('time')
        plt.ylabel('head')
        for k in range(0, nr):
            plt.plot(Time_H, Head[:, j, k], 'b')
            plt.plot(Time_H, Ref_Head[:, j], 'r-')
    plt.savefig('Head_' + str(maxIter) + '.png')
    plt.close()