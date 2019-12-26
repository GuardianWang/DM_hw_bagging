import numpy as np
import matplotlib.pyplot as plt


def get_contingency(results_a, results_b):
    """contingency table"""

    contingency_table = np.zeros((results_a.shape[0], results_b.shape[0]))
    for result_a, result_b in zip(results_a, results_b):
        contingency_table[result_a, result_b] += 1

    return contingency_table


def get_k(table):
    """k error"""

    m = table.sum()
    p1 = table.diagonal().sum() / m
    p2 = (table.sum(0) * table.sum(1)).sum() / (m ** 2)
    k = (p1 - p2) / (1 - p2)

    return k


def get_disagreement(table):
    """disagreement"""

    s = table.sum()
    agree = table.diagonal().sum()
    dis = (s - agree) / s

    return dis


def get_error(confusion_map):
    """error from a confusion map"""

    error = 1 - confusion_map.diagonal().sum() / confusion_map.sum()

    return error


def main(draw_k: bool=False):

    cms = np.load('../log/cm/cms.npy')  # (10, 17, 17)
    probs = np.load('../log/cm/probabilities.npy')  # (10, 680, 17)
    results = probs.argmax(2)  # (10, 680)
    k_values = []
    mean_errors = []
    disagreements = []
    combinations = []

    for i in range(results.shape[0] - 1):
        for j in range(i + 1, results.shape[0]):
            contingency = get_contingency(results[i], results[j])
            k = get_k(contingency)
            k_values.append(k)

            disagreement = get_disagreement(contingency)
            disagreements.append(disagreement)

            mean_error = (get_error(cms[i]) + get_error(cms[j])) / 2
            mean_errors.append(mean_error)

            combinations.append([i, j])

    if draw_k:
        plt.scatter(k_values, mean_errors, s=10)
        plt.xlabel('k')
        plt.ylabel('mean error')
        plt.xlim([0.2, 0.8])
        plt.ylim([0, 0.7])
        # plt.show()
        plt.savefig('../log/cm/k_error', dpi=200)

    disagreements = np.array(disagreements)
    combinations = np.array(combinations)
    combinations = combinations[disagreements.argsort()[::-1]]
    combinations = combinations.tolist()
    selected_classifiers = list(set(sum(combinations[:10], [])))

    print(selected_classifiers)
    print('mean of disagreement: %.3f' % disagreements.mean())
    print('std of disagreement: %.3f' % disagreements.std())


if __name__ == '__main__':
    main(draw_k=True)
