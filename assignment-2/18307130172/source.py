
import argparse

from handout import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--framework', choices=['tf', 'pt'], default='pt')
    parser.add_argument('--len', type=int, default=10)
    parser.add_argument('--cuda', action="store_true")
    arg = parser.parse_args()

    if arg.framework == 'pt':
        pt_main(arg.len, arg.cuda)
        pt_adv_main()
    elif arg.framework == 'tf':
        tf_main()
        tf_adv_main()
    else:
        raise RuntimeError
