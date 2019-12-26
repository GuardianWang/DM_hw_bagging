import numpy as np
import matplotlib.pyplot as plt

from model.config import arguments


def draw_stat(round_num, save_fig: bool=False,
              together: bool=False, fig_ax_pairs=None):
    """draw stat for each round separately"""

    base_path = '../log/round'
    train_total_loss = np.load(
        base_path + '%.2d_train_total_loss.npy' % round_num, allow_pickle=True).item()
    train_total_acc = np.load(
        base_path + '%.2d_train_total_acc.npy' % round_num, allow_pickle=True).item()

    test_total_loss = np.load(
        base_path + '%.2d_test_total_loss.npy' % round_num, allow_pickle=True).item()
    test_total_acc = np.load(
        base_path + '%.2d_test_total_acc.npy' % round_num, allow_pickle=True).item()

    train_epoch = list(train_total_loss)
    test_epoch = list(test_total_loss)

    train_loss = [train_total_loss[k] for k in train_epoch]
    train_acc = [train_total_acc[k] for k in train_epoch]
    test_loss = [test_total_loss[k] for k in test_epoch]
    test_acc = [test_total_acc[k] for k in test_epoch]

    color1, color2 = None, None
    if together:
        color1 = '#0000' + hex(round_num * 5 + 150).replace('0x', '')
        color2 = '#00' + hex(round_num * 5 + 150).replace('0x', '') + '00'

    # loss
    if together:
        fig1, ax1 = fig_ax_pairs[0]
    else:
        fig1, ax1 = plt.subplots()

    if together:
        ax1.plot(train_epoch, train_loss, color=color1)
        ax1.plot(test_epoch, test_loss, color=color2)
    else:
        ax1.plot(train_epoch, train_loss)
        ax1.plot(test_epoch, test_loss)

    ax1.legend(['train', 'test'])
    ax1.set_ylabel('loss')
    ax1.set_xlabel('epoch (non-assemble)')

    if save_fig and (not together):
        fig1.savefig('../log/loss_round_%.2d' % round_num, dpi=200)

    # accuracy
    if together:
        fig2, ax2 = fig_ax_pairs[1]
    else:
        fig2, ax2 = plt.subplots()

    if together:
        ax2.plot(train_epoch, train_acc, color=color1)
        ax2.plot(test_epoch, test_acc, color=color2)
    else:
        ax2.plot(train_epoch, train_acc)
        ax2.plot(test_epoch, test_acc)

    ax2.legend(['train', 'test'])
    ax2.set_ylim([0, 1])
    ax2.set_ylabel('accuracy')
    ax2.set_xlabel('epoch (non-assemble)')

    if save_fig and (not together):
        fig2.savefig('../log/acc_round_%.2d' % round_num, dpi=200)

    return


def draw_stat_all(round_num: int=10):
    """draw all stat in the same figure"""

    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    fig_ax_pairs = [[fig1, ax1], [fig2, ax2]]

    for round_n in range(round_num):
        draw_stat(round_num=round_n, together=True, fig_ax_pairs=fig_ax_pairs)

    fig1.savefig('../log/loss_together', dpi=200)
    fig2.savefig('../log/acc_together', dpi=200)

    return


if __name__ == '__main__':
    # argument = arguments()
    draw_stat_all()
