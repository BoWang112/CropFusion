# -*- coding: utf-8 -*-
"""
download_data.py —— 2WikiMultiHopQA 数据下载（hf-mirror 镜像）
================================================================
下载 validation 与 train parquet 到 data/。已存在则跳过。
来源：HuggingFace datadream/wiki_hop（经 hf-mirror.com 镜像）。
"""
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
DATA = os.path.join(BASE, 'data')
os.makedirs(DATA, exist_ok=True)

MIRROR = 'https://hf-mirror.com/datasets/datadream/wiki_hop/resolve/main/data'

FILES = {
    'validation.parquet': 'validation-00000-of-00001.parquet',
    'train_0.parquet': 'train-00000-of-00002.parquet',
}


def main():
    import urllib.request
    for local, remote in FILES.items():
        dst = os.path.join(DATA, local)
        if os.path.exists(dst):
            print(f'[存在] {local}（跳过）')
            continue
        url = f'{MIRROR}/{remote}'
        print(f'下载 {url} ...')
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=600) as r, open(dst, 'wb') as f:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
        print(f'  完成: {local} ({os.path.getsize(dst)/1e6:.1f} MB)')


if __name__ == '__main__':
    main()
