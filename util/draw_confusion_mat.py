import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from model.model import BaseNet
from model.config import arguments
from dataset.dataset import FlowerData


def get_cm_each_round(args, device, dataloader_test, round_num: int=None, all_classes: bool=False):
    """confusion matrix and probabilities each round"""

    network = BaseNet(num_class=args.class_num)
    if all_classes:
        network.load_state_dict(torch.load('../checkpoint/all_class.pth'))
    else:
        network.load_state_dict(torch.load(
            '../checkpoint/round%.2d_epoch%.4d.pth' % (round_num, args.epochs)))
    network = network.to(device).half()
    network.eval()

    prob = np.zeros((args.class_num * args.num_image_per_class // 2, args.class_num))
    cm = np.zeros((args.class_num, args.class_num))

    with torch.no_grad():
        for batch, (data, target) in enumerate(tqdm(dataloader_test)):
            data = data.to(device).half()
            target = target.to(device).long()
            output = network(data)

            _, pred = torch.max(output, 1)

            target = target.cpu().numpy()
            pred = pred.cpu().numpy()
            output = F.softmax(output, 1).cpu().numpy()

            idx1 = batch * args.test_batch_size
            idx2 = idx1 + args.test_batch_size
            prob[idx1: idx2, :] = output

            for i, j in zip(target, pred):
                cm[i, j] += 1

    return cm, prob


def get_confidence(cms, normalization: bool=False, save: bool=False):
    """accuracy of each classifier on each class

    normalization: weighted by precision
    normalization = False: weighted by accuracy
    """

    confidences = np.zeros((cms.shape[0], cms.shape[1]))  # (10, 17)

    for i in range(confidences.shape[0]):

        if normalization:
            cms[i] /= cms[i].sum(0)
        else:
            cms[i] /= cms[i].sum(1)

        confidences[i] = cms[i].diagonal()

    suffix = 'confidences'
    if normalization:
        suffix += '_normalized'

    if save:
        np.save('../log/cm/' + suffix, confidences)

    return confidences


def plot_cm(matrix, round_num: int=None, suffix=''):
    """draw confusion matrix"""
    classes = ['%d' % j for j in range(matrix.shape[0])]

    # Normalize by row
    matrix = matrix.astype(np.float)
    linesum = matrix.sum(1)
    linesum = np.dot(linesum.reshape(-1, 1), np.ones((1, matrix.shape[1])))
    matrix /= linesum
    # plot
    plt.switch_backend('agg')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(matrix)
    fig.colorbar(cax)
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(1))
    for i in range(matrix.shape[0]):
        ax.text(i, i, str('%.2f' % (matrix[i, i] * 100)), va='center', ha='center', fontsize=5.5)
    ax.set_xticklabels([''] + classes, rotation=90)
    ax.set_yticklabels([''] + classes)

    if round_num:
        suffix += '_round_%.2d' % round_num

    plt.savefig('../log/cm/cm%s.png' % suffix, dpi=200)

    plt.close()


def get_cm_assemble_prob(confusion_all, probs_all, confidences_all, targets, save: bool=False,
                         classifier_num=None, use_weight: bool=False, classifier_list=None,
                         normalization: bool=False):
    """
    soft vote

    cms: (10, 17, 17)
    probs: (10, 680, 17)
    confidences: (10, 17)
    targets: (680,)
    save: save confusion matrix as .npy
    classifier_num: use the first `classifier_num` classifiers to assemble a new classifier
    """
    cms = confusion_all
    probs = probs_all
    confidences = confidences_all

    if normalization:
        confidences = get_confidence(cms, normalization=normalization)

    if classifier_num:
        cms = cms[:classifier_num]
        probs = probs[:classifier_num]
        confidences = confidences[:classifier_num]
    if classifier_list:
        cms = cms[classifier_list]
        probs = probs[classifier_list]
        confidences = confidences[classifier_list]

    cm_assemble = np.zeros(cms.shape[1:])

    probs = probs.transpose((1, 0, 2))  # 680 * 10 * 17

    if use_weight:
        probs = probs * confidences  # 680 * 10 * 17

    probs = probs.sum(1)  # 680 * 17
    predictions = probs.argmax(1)

    for target, prediction in zip(targets, predictions):
        cm_assemble[int(target), prediction] += 1

    if save:
        if classifier_num:
            if use_weight:
                np.save('../log/cm/cm_assemble_prob_weight_%.2dclassifiers' % classifier_num, cm_assemble)
            else:
                np.save('../log/cm/cm_assemble_prob_%.2dclassifiers' % classifier_num, cm_assemble)

    acc = cm_assemble.diagonal().sum() / cm_assemble.sum()

    suffix = ', soft vote'
    if use_weight:
        suffix += ', use weight'
    else:
        suffix += ', no weight'

    if classifier_num:
        suffix += ', %d classifiers' % classifier_num

    if classifier_list:
        suffix += ', selected list'

    if normalization:
        suffix += ', normalization'

    print('accuracy of assemble method' + suffix + ' : %.4f' % acc)

    return cm_assemble


def get_cm_assemble_vote(confusion_all, probs_all, confidences_all, targets, save: bool=False,
                         classifier_num: int=None, use_weight: bool=False, classifier_list=None,
                         normalization: bool = False):
    """
    hard vote

    cms: (10, 17, 17)
    probs: (10, 680, 17)
    confidences: (10, 17)
    targets: (680,)
    save: save confusion matrix as .npy
    classifier_num: use the first `classifier_num` classifiers to assemble a new classifier
    """

    cms = confusion_all
    probs = probs_all
    confidences = confidences_all

    if normalization:
        confidences = get_confidence(cms, normalization=normalization)

    if classifier_num:
        cms = cms[:classifier_num]
        probs = probs[:classifier_num]
        confidences = confidences[:classifier_num]
    if classifier_list:
        cms = cms[classifier_list]
        probs = probs[classifier_list]
        confidences = confidences[classifier_list]

    cm_assemble = np.zeros(cms.shape[1:])

    probs = probs.transpose((1, 0, 2))  # 680 * 10 * 17
    probs = probs.argmax(2)  # 680 * 10, the vote of each classifier
    votes = np.zeros((probs.shape[0], cms.shape[2]))  # 680 * 17, the vote of each class

    for i in range(probs.shape[0]):
        for j in range(probs.shape[1]):
            if use_weight:
                votes[i, probs[i, j]] += confidences[j, probs[i, j]]
            else:
                votes[i, probs[i, j]] += 1

    predictions = votes.argmax(1)

    for target, prediction in zip(targets, predictions):
        cm_assemble[int(target), prediction] += 1

    if save:
        if classifier_num:
            if use_weight:
                np.save('../log/cm/cm_assemble_vote_weight_%.2dclassifiers' % classifier_num, cm_assemble)
            else:
                np.save('../log/cm/cm_assemble_vote_%.2dclassifiers' % classifier_num, cm_assemble)

    acc = cm_assemble.diagonal().sum() / cm_assemble.sum()

    suffix = ', hard vote'
    if use_weight:
        suffix += ', use weight'
    else:
        suffix += ', no weight'

    if classifier_num:
        suffix += ', %d classifiers' % classifier_num

    if classifier_list:
        suffix += ', selected list'

    if normalization:
        suffix += ', normalization'

    print('accuracy of assemble method' + suffix + ' : %.4f' % acc)

    return cm_assemble


def main(args, matrix_from_file: bool = False):
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    if not matrix_from_file:
        cms = np.zeros((10, args.class_num, args.class_num))  # (10, 17, 17)
        probs = np.zeros((10, args.class_num * args.num_image_per_class // 2, args.class_num))  # (10, 680, 17)

        dataset_test = FlowerData(args, split='test')
        dataloader_test = DataLoader(dataset_test, batch_size=args.test_batch_size,
                                     shuffle=False, num_workers=10)

        for i in range(10):
            cm, prob = get_cm_each_round(args, device, dataloader_test, round_num=i)
            cms[i], probs[i] = cm, prob

        confidences = get_confidence(cms)

        np.save('../log/cm/cms.npy', cms)
        np.save('../log/cm/probabilities.npy', probs)
    else:
        cms = np.load('../log/cm/cms.npy')
        probs = np.load('../log/cm/probabilities.npy')
        confidences = np.load('../log/cm/confidences.npy')
        targets = np.load('../log/cm/targets.npy')

    # for i in range(1, 11):
        cm = get_cm_assemble_vote(cms, probs, confidences, targets)
        plot_cm(cm, suffix='_hard_no_weight')

    #     get_cm_assemble_vote(cms, probs, confidences, targets, use_weight=True)

        cm = get_cm_assemble_vote(cms, probs, confidences, targets, use_weight=True, normalization=True)
        plot_cm(cm, suffix='_hard_weight')

        cm = get_cm_assemble_prob(cms, probs, confidences, targets)
        plot_cm(cm, suffix='_soft_no_weight')

        # get_cm_assemble_prob(cms, probs, confidences, targets, use_weight=True)

        cm = get_cm_assemble_prob(cms, probs, confidences, targets, use_weight=True, normalization=True)
        plot_cm(cm, suffix='_soft_weight')

    # for i in range(10):
    #     # plot confusion matrix
    #     plot_cm(cms[i], round_num=i)


if __name__ == '__main__':
    argument = arguments()
    main(argument, matrix_from_file=True)

    # args = argument
    # use_cuda = not argument.no_cuda and torch.cuda.is_available()
    # device = torch.device("cuda:0" if use_cuda else "cpu")
    # dataset_test = FlowerData(args, split='test')
    # dataloader_test = DataLoader(dataset_test, batch_size=args.test_batch_size,
    #                              shuffle=False, num_workers=10)
    # cm, _ = get_cm_each_round(args, device, dataloader_test, all_classes=True)
    # plot_cm(cm, suffix='all_classes')
    # print(cm.diagonal().sum() / cm.sum())
