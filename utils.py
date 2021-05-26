import numpy as np
import torch.nn as nn
import torch

def MSE_missing(reconstruction, orig_sample, mask):
    """
    Calculates MSE for area with missing data

    :param reconstruction: VAE reconstruction
    :param orig_sample: Original data sample
    :param mask: Mask
    :return: MSE loss for missing data area
    """

    #i
    # Checking performance on missing
    inverse_mask = np.logical_not(mask)

    # Applying inverse mask to look at previous masked area
    missing_imputed = reconstruction.detach() * inverse_mask.float()
    missing_true = orig_sample.detach() * inverse_mask.float()

    #Calculate MSE
    masked_area = np.logical_not(mask[0]).sum()  # size of masked area
    #print(masked_area)
    mse = torch.mean(np.square(missing_true - missing_imputed), dim=1) / masked_area  # MSE of masked area

    return mse.mean().item()

class PrintLayer(nn.Module):
    def __init__(self):
        super(PrintLayer, self).__init__()

    def forward(self, x):
        # Do your print / debug stuff here
        #print(x.shape)
        return x
    
class EarlyStop:
    def __init__(self, steps=8, eps=0, path='checkpoint.pt'):
        self.counter = 0
        self.eps = eps
        self.path = path
        self.steps = steps
        self.opt_score = None
        self.early_stop = False
        
    def __call__(self, loss, model):

        score = -loss
        
        if self.opt_score is None:
            self.opt_score = score
            torch.save(model.state_dict(), self.path)
        elif score < self.opt_score + self.eps:
            self.counter += 1
            if self.counter >= self.steps:
                self.early_stop = True
        else:
            self.opt_score = score
            torch.save(model.state_dict(), self.path)
            self.counter = 0


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class UnFlatten(nn.Module):
    def forward(self, input):
        dims = input.size(0)
        return input.view(dims, 32, 7, 7)


class PrintLayer(nn.Module):
    def __init__(self):
        super(PrintLayer, self).__init__()

    def forward(self, x):
        print(x.size())
        return x


def majority_vote(xm):
    # majority voting only works for binarized xm. Unless threshold is specified (e.g. 0.5 for vote 1 and 0)
    # first count number of samples larger than 0 and 0
    vote_1=(xm>0).sum(0)
    vote_0=(xm==0).sum(0)
    # find where the count of 1s is larger than 0
    xm=(vote_1>vote_0)
    #impute 1 on where the count is larger and 0 on where it is less
    xm=xm*1
    return xm


def sample_mean(xm):
    return xm.mean(0)


