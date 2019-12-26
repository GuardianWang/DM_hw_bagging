import argparse


def arguments():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch 17Flowers')

    parser.add_argument('--batch-size', type=int, default=20, metavar='N',
                        help='input batch size for training (default: 4)')
    parser.add_argument('--test-batch-size', type=int, default=20, metavar='N',
                        help='input batch size for testing (default: 4)')
    parser.add_argument('--epochs', type=int, default=80, metavar='N',
                        help='number of epochs to train')
    parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--test-interval', type=int, default=1, metavar='N',
                        help='how many batches to wait before testing')
    parser.add_argument('--save-interval', type=int, default=80,
                        help='For Saving the current Model')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum (default: 0.9)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--data-path', default=r'/17flowers',
                        help='data path')
    parser.add_argument('--image-height', type=int, default=64,
                        help='image height')
    parser.add_argument('--image-width', type=int, default=64,
                        help='image width')
    parser.add_argument('--class-num', type=int, default=17,
                        help='number of classes')
    parser.add_argument('--main-class', type=int, default=0,
                        help='main class `k` of single classifier, '
                             '(0 <= k <= 16) to specify a main class')
    parser.add_argument('--test-split', default='test_small',  # 'test' or 'test_small'
                        help='test_small: 80 positive and 80 negative'
                             'test: use all images')
    parser.add_argument('--num-image-per-class', type=int, default=80,
                        help='number of images per class')
    parser.add_argument('--num-image-per-negative', type=int, default=5,
                        help='number of images per negative class')
    parser.add_argument('--weight', type=float, default=1.5,
                        help='weight parameter')
    args = parser.parse_args()

    return args
