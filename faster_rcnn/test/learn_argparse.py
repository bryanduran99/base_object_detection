import argparse

# https://docs.python.org/zh-cn/3/howto/argparse.html
if __name__ =='__main__':
    parser = argparse.ArgumentParser(
        description=__doc__)

    parser.add_argument('must',default=0,help = '')
    parser.add_argument('--a', default='./', help='dataset')
    parser.add_argument('--b', default=False, help='dataset')
    args = parser.parse_args()
    print(args)