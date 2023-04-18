import os
import sys
import random
import numpy as np

import mindspore as ms
from tools import train_net, test_net
from utils.parser import get_args
#from utils.goat_utils import setup_env, init_seed
#from mmengine.dist import is_main_process


def main():
    print(ms.communication.get_group_size())
    '''torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = True'''
    args = get_args()
    if args.use_bp:
        args.qk_dim = 768
    else:
        args.qk_dim = 1024
    args.benchmark = 'FineDiving'
    '''if is_main_process():
        print(args)'''
    print(args)

    if args.launcher == 'none':
        args.distributed = False
    else:
        raise NotImplementedError()
        args.distributed = True

    setup_env(args.launcher, distributed=args.distributed)
    init_seed(args)

    if args.test:
        test_net(args)
    else:
        train_net(args)

def init_seed(args):
    """
    Set random seed for torch and numpy.
    """
    random.seed(args.seed)
    np.random.seed(args.seed)
    ms.set_seed(args.seed)

    '''torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False'''


if __name__ == '__main__':
    main()
