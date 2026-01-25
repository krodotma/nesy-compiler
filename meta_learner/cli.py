#!/usr/bin/env python3
import argparse
from .learner import MetaLearner

def main():
    p = argparse.ArgumentParser()
    p.add_argument('cmd', choices=['update', 'list-skills'], help='MetaLearner command')
    args = p.parse_args()
    ml = MetaLearner()
    if args.cmd == 'update':
        ml.update()
        print('MetaLearner model updated.')
    elif args.cmd == 'list-skills':
        print('\n'.join(ml.registry.list()))

if __name__ == '__main__':
    main()
