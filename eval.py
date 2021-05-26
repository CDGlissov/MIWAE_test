import torch
import numpy as np
import matplotlib.pyplot as plt
import data_Loader
import model
import os
from tqdm import tqdm
import torch.distributions as td
#metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from scipy.stats import sem
import pickle


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 100
root = os.getcwd()
path = root+'/model results/studentt/'
net = model.VAE(z_dim=50, bs=batch_size, encoder_type='studentt').to(device)

X_train, X_test = data_Loader.mnist_numpy(labels=False, data_type="mnist", normalize=False, binarize=True)
mask_train, mask_test = data_Loader.data_corruption(X_train, X_test, mode='snp', percentage=0.5,
                                                     block_size=0)  # All train data is not used
test_data = torch.utils.data.TensorDataset(torch.Tensor(X_test), torch.Tensor(mask_test))
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)


x, mask=next(iter(test_loader))
x=x.view(batch_size, 1, 28, 28)
mask=mask.view(batch_size, 1, 28, 28)
xhat = x*mask

net.load_state_dict(torch.load(path + 'checkpoint' + str(2) + '.pt.tar'))

xm_list = []
x_prob_list = []
impute_type="multiple"
for i in range(5):
    if impute_type=="single":
        xm, x_prob=net.impute(xhat[i,:,:,:].unsqueeze(0), K=10000, mask=mask[i,:,:,:].unsqueeze(0), impute_type=impute_type)
    else:
        xm, x_prob = net.impute(xhat[i, :, :, :].unsqueeze(0), K=10000, mask=mask[i, :, :, :].unsqueeze(0), S=20,
                                impute_type=impute_type)
    xm_list.append(xm)
    x_prob_list.append(x_prob)

    xhat[i,:,:,:][mask[i,:,:,:]==0] = xm.view(1,28,28)[mask[i,:,:,:]==0]

fig, ax = plt.subplots(5, 4)
for i in range(5):
    incomplete_samp=x[i] * mask[i]
    incomplete_samp[mask[i, :, :, :] == 0]=0.5
    ax[i, 0].imshow(x[i].detach().cpu().view(28, 28), cmap='gray', vmin=0, vmax=1, aspect='auto')
    ax[i, 1].imshow(incomplete_samp.detach().cpu().view(28, 28), cmap='gray', vmin=0, vmax=1, aspect='auto')
    ax[i, 2].imshow(xhat[i].detach().cpu().view(28,28), cmap='gray', vmin=0, vmax=1, aspect='auto')
    ax[i, 3].imshow(x_prob_list[i].detach().cpu().view(28,28), cmap='gray', aspect='auto')
    for j in range(4):
        ax[i, j].axis('off')
ax[0,0].set_title("Complete")
ax[0,1].set_title("Incomplete")
ax[0,2].set_title("Imputed")
ax[0,3].set_title("Probability")
plt.subplots_adjust(wspace=0, hspace=0)
#plt.show()

path = root+'/model results/Gaussian/'

def metrics_models(impute_type, models = 1):
    mean_accuracy = []
    mean_error_ce = []
    for i in range(models):
        print("model " + str(i))
        net.load_state_dict(torch.load(path + 'checkpoint' + str(i) + '.pt.tar'))
        net.eval()
        error_accuracy = []
        error_ce = []
        for x_obs in tqdm(test_loader):
            x=x_obs[0].view(batch_size, 1, 28, 28)
            mask = x_obs[1].view(batch_size, 1, 28, 28)
            xhat=x*mask

            for i in range(batch_size):
                sample_batch = xhat[i,:,:,:]
                sample_mask = mask[i,:,:,:]

                if impute_type == "single":
                    xm, x_prob = net.impute(sample_batch.unsqueeze(0), K=1000, mask=sample_mask.unsqueeze(0), impute_type=impute_type)
                else:
                    xm, x_prob = net.impute(sample_batch.unsqueeze(0), K=1000, mask=sample_mask.unsqueeze(0), S=20, impute_type=impute_type)
                sample_batch[sample_mask==0] = xm.view(1,28,28)[sample_mask==0]
                sample_batch=sample_batch.view(-1,784)
                x_batch=x[i].view(-1,784)
                sample_mask=sample_mask.view(-1,784)
                err_accuracy=accuracy_score(x_batch[sample_mask==0], sample_batch[sample_mask==0])
                err_ce = log_loss(x_batch[sample_mask==0], sample_batch[sample_mask==0].clamp(1e-8,1-1e-7));
                error_accuracy.append(err_accuracy)
                error_ce.append(err_ce)
        mean_accuracy.append(np.mean(error_accuracy))
        mean_error_ce.append(np.mean(error_ce))

    std_error_accuracy = sem(mean_accuracy)
    std_error_ce = sem(mean_error_ce)
    mean_acc = np.mean(mean_accuracy)
    mean_ce = np.mean(mean_error_ce)

    return std_error_accuracy, std_error_ce, mean_acc, mean_ce

std_acc, std_ce, mean_acc, mean_ce = metrics_models(impute_type="multiple", models = 5)

stats=[std_acc, std_ce, mean_acc, mean_ce]

with open('stats_single_impute.p', 'wb') as fp:
    pickle.dump(stats, fp)

print(std_acc)
print(std_ce)
print(mean_acc)
print(mean_ce)

