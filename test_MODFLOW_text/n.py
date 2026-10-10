
import numpy as np
import flopy.utils.binaryfile as bf
import matplotlib.pyplot as plt

#Train_Data = np.load(r'Datasets3.npy').astype('float32')# the path of training data
#niter = int(Train_Data.shape[0]/opt.batch_size)
#Train_Data = Train_Data[0:1,:,:,:]
#Train_Data = Train_Data.reshape(129,129)
Train_Data = np.loadtxt('ref')

Head_file = r'test.hed'

h_n_ob = 25  # the total number of head observation points
h_spacing = 30 # the spacing between two wells
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



#ln_K_ref = np.loadtxt(r'D:\SDSMT\Paper_VAE_GAN_Mathematical Geosciences\Experiments\ES_MDA\Ref\Continuous_6_log.txt')
ln_K_ref = Train_Data
plt.matshow(ln_K_ref)
plt.colorbar()
for i in range(1, h_n_ob + 1):  # the location of wells
    plt.scatter(eval('Head_' + str(i) + '_x'), eval('Head_' + str(i) + '_y'), marker='o', c='r', edgecolors='r', s=20)

plt.savefig('Coordinate_Head.png')
plt.close()

for i in range(1, h_n_ob + 1):
    locals()['idx_' + str(i)] = (layer, eval('Head_' + str(i) + '_x'), eval('Head_' + str(i) + '_y'))

headobj = bf.HeadFile(Head_file)
Time = headobj.get_ts(idx_1)[:, 0]
Len_T = len(Time)
with open('Ref_Time_Head.txt', 'w') as f:
    for i in range(Len_T):
        f.write(str(Time[i]))
        f.write('\n')
f.close()


ts = np.ones((Len_T * h_n_ob, 2), dtype=np.float32)
tss = np.ones((Len_T * h_n_ob, 1), dtype=np.float32)
for t in range(1, h_n_ob + 1):
    locals()['ts_' + str(t)] = headobj.get_ts(eval('idx_' + str(t)))
    ts[(t - 1) * len(Time):t * len(Time), :] = eval('ts_' + str(t))
    tss = ts[:, 1]

with open('Ref_Head.txt', 'a') as f:
    for i in range(len(tss)):
        f.write(str(tss[i]))
        f.write('\n')
f.close()
# Configure data loader
#Train_Data = np.load(r'Datasets3.npy').astype('float32')# the path of training data
#niter = int(Train_Data.shape[0]/opt.batch_size)
#Train_Data = Train_Data[0:1,:,:,:]
#Train_Data = Train_Data.reshape(129,129)
#plt.matshow(Train_Data)
#plt.show()
