#!/usr/bin/env python3
import argparse
from pathlib import Path

import wfdb


def parse_args():
    parser = argparse.ArgumentParser(
        description='Download LUDB to a local target directory.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='../public_dataset/physionet.org/files/ludb/1.0.1/data',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    target_dir = Path(args.output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    wfdb.dl_database('ludb', dl_dir=str(target_dir))


if __name__ == '__main__':
    main()
