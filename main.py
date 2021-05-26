# MIAWE
# Johan Ye & Christian Glissov

import torch
import torch.optim as optim
import argparse
import numpy as np
import data_Loader
import utils
import model
import matplotlib.pyplot as plt
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--encoder', type=str, default='gaussian')
parser.add_argument('--testing', type=int, default=0)
args = parser.parse_args()

testing = args.testing

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)

# Data loading
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Normalized with mean=0.5 and std = 0.5
X_train, X_test = data_Loader.mnist_numpy(labels=False, data_type="mnist", normalize=False, binarize=True)
train_idx = np.random.choice(np.arange(60000), size=55000, replace=False)
val_idx = np.array(list(set(np.arange(60000)).symmetric_difference(train_idx)))

# block or SnP (retarded implementation, but too lazy to fix function)
mask_train, mask_test = data_Loader.data_corruption(X_train[train_idx], X_test, mode='snp', percentage=0.5,
                                                    block_size=0)
mask_val, mask_test = data_Loader.data_corruption(X_train[val_idx], X_test, mode='snp', percentage=0.5, block_size=0)
batch_size = 200  # has to be number that 60k is divisible by due to lazy coding

# Creates dataset with both data and mask
train_data = torch.utils.data.TensorDataset(torch.Tensor(X_train[train_idx]), torch.Tensor(mask_train))
val_data = torch.utils.data.TensorDataset(torch.Tensor(X_train[val_idx]), torch.Tensor(mask_val))
test_data = torch.utils.data.TensorDataset(torch.Tensor(X_test), torch.Tensor(mask_test))

# Data loader including data and mask
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)

# FOR DEBUGGING VAE CLASS.
net = model.VAE(z_dim=50, bs=batch_size, encoder_type=args.encoder).to(device)
optimizer = optim.Adam(net.parameters(), lr=2e-4)

# Set early stop and path for saving
early_stopping = utils.EarlyStop(steps=50, path='checkpoint' + str(args.seed) + '.pt.tar')

file1 = 'output' + str(args.seed) + '_' + str(args.encoder) + '.txt'
outfile = open(file1, 'w')
outfile.write('Epoch\tTrain Loss\tValidation loss\n')

n_epoch = 3 if testing == True else 250
i = 0
beta = 0.0002
train_loss, train_kl, train_nll = [], [], []
valid_loss, valid_kl, valid_nll = [], [], []

n_samples = 50
print(n_samples)

for epoch in range(n_epoch):
    batch_loss, batch_kl, batch_nll = [], [], []
    net.train()

    for x_obs in tqdm(train_loader):
        x = x_obs[0].view(x_obs[0].shape[0], 1, 28, 28).to(device)
        mask = x_obs[1].view(x_obs[0].shape[0], 1, 28, 28).to(device)
        xhat = x * mask
        z, loss, mu_enc, std_enc, mu_dec, kl, nll = net(xhat, n_samples, beta=1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_loss.append(loss.item())
        batch_kl.append(kl.item())
        batch_nll.append(nll.item())
        i += 1

    Mean_Train_Loss = np.mean(batch_loss)
    train_loss.append(Mean_Train_Loss)
    train_kl.append(np.mean(batch_kl))
    train_nll.append(np.mean(batch_nll))
    if i > 4500 and beta < 1:  # 15 epoch warm-up
        beta += 0.0002  # 15 epochs again

    batch_loss = []

    with torch.no_grad():
        net.eval()
        for x_obs in tqdm(val_loader):
            x = x_obs[0].view(x_obs[0].shape[0], 1, 28, 28).to(device)
            mask = x_obs[1].view(x_obs[0].shape[0], 1, 28, 28).to(device)
            xhat = x * mask
            z, loss, mu_enc, std_enc, mu_dec, kl, nll = net(xhat, n_samples)
            batch_loss.append(loss.item())
            batch_kl.append(kl.item())
            batch_nll.append(nll.item())
        Mean_Valid_Loss = np.mean(batch_loss)
        valid_loss.append(Mean_Valid_Loss)
        valid_kl.append(np.mean(batch_kl))
        valid_nll.append(np.mean(batch_nll))

    print("Epoch:" + str(epoch) + " Train LOSS:" + str(Mean_Train_Loss) + " Validation LOSS:" + str(Mean_Valid_Loss))

    outfile.write("Epoch:" + str(i) +
                  " Train LOSS:" + str(Mean_Train_Loss) +
                  " Validation LOSS:" + str(Mean_Valid_Loss) + '\n')

    early_stopping(Mean_Valid_Loss, net)
    if early_stopping.early_stop:
        print("Early stopping")
        break

for i in net.parameters():
    i.requires_grad = False

for i in net.enc.parameters():
    i.requires_grad = True

fig, ax = plt.subplots(1, 3, figsize=(10, 5))
x_val = np.arange(n_epoch)
ax[0].plot(x_val, train_loss, label='Train ELBO')
ax[0].plot(x_val, valid_loss, label='Valid ELBO')
ax[0].set_title('ELBO Loss Curves')

ax[1].plot(x_val, train_kl, label='Train KL')
ax[1].plot(x_val, valid_kl, label='Valid KL')
ax[1].set_title('KL Training Curve')

ax[2].plot(x_val, train_nll, label='Train NLL')
ax[2].plot(x_val, valid_nll, label='Valid NLL')
ax[2].set_title('kl Training Curve')
plt.legend(loc='best')
plt.savefig('Plotting' + str(args.seed) + '.pdf')

print('network training done')
print('proceeding encoder training')
test_optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=1e-3)

n_epoch = 3 if testing == True else 100
i = 0
test_loss, test_kl, test_nll = [], [], []
n_samples = 50

# Set early stop and path for saving
early_stopping2 = utils.EarlyStop(steps=50, path='checkpoint' + str(args.seed) + '_encoder.pt.tar')

for epoch in range(n_epoch):
    net.train()
    batch_loss = []
    for x_obs in tqdm(test_loader):
        x = x_obs[0].view(x_obs[0].shape[0], 1, 28, 28).to(device)
        mask = x_obs[1].view(x_obs[0].shape[0], 1, 28, 28).to(device)
        xhat = x * mask

        z, loss, mu_enc, std_enc, mu_dec, kl, nll = net(xhat, n_samples)
        test_optimizer.zero_grad()
        loss.backward()
        test_optimizer.step()
        batch_loss.append(loss.item())
        batch_kl.append(kl.item())
        batch_nll.append(nll.item())

    Mean_test_Loss = np.mean(batch_loss)
    test_loss.append(Mean_test_Loss)
    test_kl.append(np.mean(batch_kl))
    test_nll.append(np.mean(batch_nll))

    outfile.write("test LOSS:" + str(Mean_test_Loss) + '\n')
    outfile.write("test LOSS:" + str(Mean_test_Loss) + '\n')

    early_stopping2(Mean_test_Loss, net)
    if early_stopping.early_stop:
        print("Early stopping")
        break

fig, ax = plt.subplots(1, 3, figsize=(10, 5))
x_val = np.arange(n_epoch)
ax[0].plot(x_val, test_loss, label='Test ELBO')
ax[0].set_title('ELBO Loss Curves')

ax[1].plot(x_val, test_kl, label='Test KL')
ax[1].set_title('KL Training Curve')

ax[2].plot(x_val, test_nll, label='Test nll')
ax[2].set_title('NLL Training Curve')
plt.legend(loc='best')
plt.savefig('Plotting_test' + str(args.seed) + '.pdf')

# This should probably be checked on the test set instead.
outfile.write("Final lower marginal likelihood bound, p(x): %g" % (-np.log(n_samples) - loss))
net.load_state_dict(torch.load('checkpoint' + str(args.seed) + '.pt.tar'))
net.eval()
