import mmap
import os
import requests
from tqdm import tqdm
from hashlib import sha1


# def download_file(url, fp):
#     r = requests.get(url)
#     assert r.status_code == 200, f'Error! Download failure. URL: {url} OUT: {fp}'
#     with open(fp, "wb") as f:
#         f.write(r.content)


def sha1_check(fp, sha1_code):
    f = open(fp, 'rb')
    sha1_obj = sha1(mmap.mmap(f.fileno(), os.path.getsize(fp), access=mmap.ACCESS_READ))
    f.close()
    return sha1_obj.hexdigest() == sha1_code.lower()


def download_file(url, fp, sha1_code=None):
    '''
    支持断点续传的下载函数
    :param url:
    :param fp:
    :param sha1_code:
    :return:
    '''

    r = requests.get(url, stream=True)

    if 'Content-Length' in r.headers:
        size = int(r.headers['Content-Length'])
        assert size >= 0, 'Error! Bad file size.'
    else:
        size = None

    r.close()

    if size is None and os.path.isfile(fp):
        # 目标不支持没有已知的大小，删掉现有的然后重新下载
        print('Info! The downloaded file has no explicit size. Will delete and download it again.')
        os.unlink(fp)

    downloaded_size = 0
    if os.path.isfile(fp):
        downloaded_size = os.path.getsize(fp)

    headers = {}
    if downloaded_size != 0 and size is not None:
        if downloaded_size == size:
            # 目标已下载完成
            if sha1_code is not None:
                if sha1_check(fp, sha1_code):
                    return
                else:
                    print('Info! The downloaded file is corrupt, download it again.')
                    os.unlink(fp)
                    downloaded_size = 0
        headers.update(dict(Range=f'bytes={downloaded_size}-'))

    proxies = {}
    if False:
        proxies.update({
            'http': 'socks5://127.0.0.1:1080',
            'https': 'socks5://127.0.0.1:1080',
        })

    try:
        r = requests.get(url, stream=True, headers=headers, proxies=proxies)
        with open(fp, 'ab') as f:
            for chunk in tqdm(r.iter_content(chunk_size=1024**2)):
                downloaded_size += len(chunk)
                f.write(chunk)
                f.flush()
    except Exception as e:
        print(e)
        raise f'Error! Download failure. URL: {url} OUT: {fp}'

    if sha1_code is not None and not sha1_check(fp, sha1_code):
        raise AssertionError('Error! The downloaded file is corrupt, please download it again.')
