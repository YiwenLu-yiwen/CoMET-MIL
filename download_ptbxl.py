#!/usr/bin/env python3
import argparse
import ast
import csv
import socket
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import urllib.error
import urllib.parse
import urllib.request


BASE_URL = 'https://physionet.org/files/ptb-xl/1.0.3/'
METADATA_FILES = ('ptbxl_database.csv', 'scp_statements.csv')


def download_file(url, dest_path, timeout=60, retries=4, retry_delay=2.0):
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists():
        return
    last_error = None
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response, open(dest_path, 'wb') as out:
                out.write(response.read())
            return
        except (TimeoutError, socket.timeout, urllib.error.URLError) as exc:
            last_error = exc
            if dest_path.exists():
                dest_path.unlink()
            if attempt == retries:
                break
            time.sleep(retry_delay * (attempt + 1))
    raise RuntimeError(f'Failed to download {url} after {retries + 1} attempts') from last_error


def download_metadata(target_root, timeout, retries, retry_delay):
    for filename in METADATA_FILES:
        download_file(
            urllib.parse.urljoin(BASE_URL, filename),
            target_root / filename,
            timeout=timeout,
            retries=retries,
            retry_delay=retry_delay,
        )


def load_ptbxl_rows(metadata_csv):
    with open(metadata_csv, newline='') as f:
        reader = csv.DictReader(f)
        return list(reader)


def select_records(rows, codes):
    selected = []
    codes = set(codes)
    for row in rows:
        scp_codes = ast.literal_eval(row['scp_codes'])
        if any(code in scp_codes for code in codes):
            selected.append(row)
    return selected


def main():
    parser = argparse.ArgumentParser(
        description='Download PTB-XL metadata or a code-filtered records100 subset.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--metadata-only', action='store_true')
    parser.add_argument(
        '--codes',
        nargs='+',
        default=['AFIB', 'AFLT', 'SR'],
        help='SCP codes to keep when downloading a filtered records100 subset',
    )
    parser.add_argument(
        '--resolutions',
        nargs='+',
        choices=['100', '500'],
        default=['100'],
        help='PTB-XL waveform resolutions to download',
    )
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--timeout', type=float, default=60.0)
    parser.add_argument('--retries', type=int, default=4)
    parser.add_argument('--retry-delay', type=float, default=2.0)
    args = parser.parse_args()

    target_root = Path(args.output_dir)
    target_root.mkdir(parents=True, exist_ok=True)
    download_metadata(target_root, args.timeout, args.retries, args.retry_delay)

    if args.metadata_only:
        return

    rows = load_ptbxl_rows(target_root / 'ptbxl_database.csv')
    selected_rows = select_records(rows, args.codes)
    if not selected_rows:
        raise RuntimeError(f'No PTB-XL records found for codes={args.codes!r}')

    jobs = []
    for row in selected_rows:
        rel_paths = []
        if '100' in args.resolutions:
            rel_paths.append(row['filename_lr'])
        if '500' in args.resolutions:
            rel_paths.append(row['filename_hr'])
        for rel_path in rel_paths:
            for suffix in ('.hea', '.dat'):
                rel_file = f'{rel_path}{suffix}'
                jobs.append((urllib.parse.urljoin(BASE_URL, rel_file), target_root / rel_file))

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        list(
            executor.map(
                lambda job: download_file(
                    *job,
                    timeout=args.timeout,
                    retries=args.retries,
                    retry_delay=args.retry_delay,
                ),
                jobs,
            )
        )


if __name__ == '__main__':
    main()
