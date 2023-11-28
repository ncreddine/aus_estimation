import torch
import os
import wandb
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import seaborn as sns
import pandas as pd
import matplotlib
import provider



def error_per_au_per_intensity(predictions, labels):
    action_units = ['au1', 'au2', 'au4', 'au5', 'au6', 'au9', 'au12', 'au15', 'au17', 'au20', 'au25', 'au26']
    mse_matrix = np.zeros(shape=(6, len(action_units)))
    for au_index, au in enumerate(action_units) :
        pred_per_au = predictions[:, au_index]
        labels_per_au = labels[:, au_index]
        mse_per_au = []
        for intensity in range(6) :
            mask = torch.argwhere(labels.cpu() == intensity)
            mse = torch.square(labels_per_au[mask].cpu() - pred_per_au[mask].cpu()).mean()
            mse_matrix[intensity, au_index] = 20.0 if torch.isnan(mse) else mse
    
    return mse_matrix, action_units, np.arange(6)

    # return pd.DataFrame( total_dataframe, columns=['AU', "Intensity", "Error"]).pivot(index="Intensity", columns="AU", values="Error")

def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw=None, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im = None, past_im = None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """


    # Normalize the threshold to the images color range.

    data = im.get_array()
    past_data = past_im.get_array()

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(data[i, j] - past_data[i, j] < 0)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)
    return texts

class Trainer :
    def __init__(self, model : torch.nn.Module, optimizer : torch.optim.Optimizer, scheduler : torch.optim.lr_scheduler._LRScheduler, 
                loss_fn : torch.nn.Module, device : int, train_loader : DataLoader, test_loader : DataLoader, resume : str, args : None ):
        self.device = device
        # torch.cuda.empty_cache()
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        ## Train dataloader
        self.train_loader = train_loader
        self.test_loader = test_loader
        ## Steps
        self.train_step = 0
        self.test_step = 0
        self.start_epoch = 0
        self.epochs = args.num_epochs
        ## Wandb logger
        self.logger = wandb.init(project = args.wandb_project_name, group=str(args.pid))
        wandb.watch(self.model, log="all")


    def train(self, epoch):
        print('\nEpoch : %d'%epoch)
        
        self.model.train()
        
        running_loss=0

        for _, facemesh, actionunits in tqdm(self.train_loader):

            
            # facemesh = facemesh.transpose(1, 2)
            # if self.args.model == 'pointnet' :
            facemesh = facemesh.data.numpy()
            # facemesh = provider.random_point_dropout(facemesh, max_dropout_ratio=0.4)
            facemesh[:, :, 0:3] = provider.random_scale_point_cloud(facemesh[:, :, 0:3])
            facemesh[:, :, 0:3] = provider.shift_point_cloud(facemesh[:, :, 0:3])
            facemesh = torch.Tensor(facemesh)
            facemesh = facemesh.transpose(2, 1)

            facemesh = facemesh.to(self.device)

            actionunits = actionunits.to(self.device)

            outputs, trans_feat  = self.model(facemesh)
            
            loss = self.loss_fn(outputs, actionunits, trans_feat)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()

        train_loss=running_loss/len(self.train_loader)
        if self.scheduler is not None : self.scheduler.step()
            
        print(f"Train Loss: {train_loss}")
        return train_loss

    def test(self, epoch) :
        self.model.eval()
        
        running_loss = 0
        total = 0
        
        y_true = []
        y_pred = []

        with torch.no_grad():
            for _, facemesh, actionunits in tqdm(self.test_loader):

                # facemesh = facemesh.data.numpy()
                # facemesh = provider.random_point_dropout(facemesh, max_dropout_ratio=0.2)
                # facemesh[:, :, 0:3] = provider.random_scale_point_cloud(facemesh[:, :, 0:3])
                # facemesh[:, :, 0:3] = provider.shift_point_cloud(facemesh[:, :, 0:3])
                # facemesh = torch.Tensor(facemesh)
                facemesh = facemesh.transpose(2, 1)

                facemesh = facemesh.to(self.device)

                actionunits = actionunits.to(self.device)

                outputs, trans_feat  = self.model(facemesh)
                
                loss = self.loss_fn(outputs, actionunits, trans_feat)

                # facemesh = facemesh.to(self.device)
                # facemesh = facemesh.transpose(1, 2)
                # actionunits = actionunits.to(self.device)

                # outputs = self.model(facemesh)

                # loss = self.loss_fn(outputs, actionunits)
                running_loss+=loss.item()
        
                total += actionunits.size(0)

                y_true.append(actionunits)
                y_pred.append(outputs)

        labels = torch.cat(y_true, 0)
        predictions = torch.cat(y_pred, 0)

        mse_per_au_per_intensity, _action_units, _intensity = error_per_au_per_intensity(predictions, labels)


        fig, ax = plt.subplots(figsize = (12, 5))
        # ax = sns.heatmap(mse_per_au_per_intensity, annot=True, fmt=".2f", ax = ax)
        im, _ = heatmap(mse_per_au_per_intensity, row_labels=_intensity, col_labels=_action_units, ax = ax, vmin= 0, vmax = 10, cbarlabel = "Mean Square Error",cmap="Wistia" )
        if epoch == 0 :
            annotate_heatmap(im = im, past_im=im,  size=8, fontweight="bold", textcolors=("black", "black"))
        else : 
            annotate_heatmap(im = im, past_im=self.prev_im,  size=9, fontweight="bold", textcolors=("red", "green"))
        self.prev_im = im

        plt.close()

        # if epoch > 0 :
        #     annotate_heatmap(ax, self.past_mse_per_au_per_intensity, textcolors=("red", "green"))

        # self.past_mse_per_au_per_intensity = ax
        

        test_loss=running_loss/len(self.test_loader)

        
        print(f"Test Loss: {test_loss}")
        return test_loss, ax 

    def start_training(self):
        for epoch in range(self.start_epoch, self.epochs):
            train_loss= self.train(epoch)
            test_loss, ax_mse =  self.test(epoch)
            logs = dict()
            logs['epochs/train_loss'] = train_loss
            logs['epochs/test_loss'] = test_loss
            logs['epochs/epoch'] = epoch
            logs['epochs/lr'] = self.optimizer.param_groups[0]['lr']
            logs['epochs/mse_per_intensity'] = wandb.Image(ax_mse)
            self.logger.log(logs)