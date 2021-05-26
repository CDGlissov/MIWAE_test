import torch
import model
import data_Loader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 200
path = '../model results/studentt/'
net = model.VAE(z_dim=50, bs=batch_size, encoder_type='studentt').to(device)


dfs = np.zeros((5, 10000, 50))

for i in range(5):
    np.random.seed(i)
    torch.manual_seed(i)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(i)

    X_train, X_test = data_Loader.mnist_numpy(labels=False, data_type="mnist", normalize=False, binarize=True)
    mask_train, mask_test = data_Loader.data_corruption(X_train, X_test, mode='snp', percentage=0.5,
                                                        block_size=0)  # All train data is not used
    test_data = torch.utils.data.TensorDataset(torch.Tensor(X_test), torch.Tensor(mask_test))
    net.load_state_dict(torch.load(path + 'checkpoint' + str(i) + '.pt.tar'))
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)

    for j, x_obs in enumerate(tqdm(test_loader)):
        x = x_obs[0].view(x_obs[0].shape[0], 1, 28, 28).to(device)
        mask = x_obs[1].view(x_obs[0].shape[0], 1, 28, 28).to(device)
        xhat = x * mask

        df = net.get_df(x)

        dfs[i, j*batch_size: (j+1)*batch_size, :] = df.cpu().detach().numpy()


z_dim = np.arange(50)
for i in range(5):
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.boxplot(dfs[i, :, :], showfliers=False)
    ax.set_xlabel('Latent dimension')
    plt.title('Plot of student-t degrees of freedom')
    plt.savefig('../model results/studentt/df_plot_' + str(i) + '.pdf', bbox_inches='tight')
