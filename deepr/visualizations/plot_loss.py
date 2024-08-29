import numpy as np
import matplotlib.pyplot as plt

from deepr.utilities.logger import get_logger

logger = get_logger(__name__)

def plot_loss(loss, stat, num_epochs, output_dir):
    loss = np.array(loss)
    plt.plot(loss, c='r')
    plt.xlim(0, num_epochs)
    plt.title(stat)
    plt.xlabel('Epoch number')
    plt.savefig(output_dir + stat + f'{len(loss):04d}' + '.png')
    plt.cla()