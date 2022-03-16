import torch
import pickle
import os
# import PIL.Image as Image
import cv2
from tqdm import tqdm
from threading import Thread
from queue import Queue


def writer_run(q: Queue):
    while True:
        r = q.get()
        if r is None:
            break
        fp, im = r
        cv2.imwrite(fp, im[:, :, ::-1])


if __name__ == '__main__':

    q = Queue(128)

    cur_dir = os.path.dirname(__file__)
    weight_path = cur_dir + '/stylegan3-t-ffhqu-256x256-shim.pkl'

    sample_dir_1 = cur_dir + '/sample_1'
    sample_dir_2 = cur_dir + '/sample_2'

    n_sample = 10000
    batch_size = 32
    device = 'cuda:0'

    gnet = pickle.load(open(weight_path, 'rb'))['G_ema']

    gnet.to(device)

    writer_thread = Thread(target=writer_run, args=(q,))
    writer_thread.start()

    with torch.inference_mode():
        for cur_i in tqdm(range(0, n_sample, batch_size)):
            cur_batch_size = min(batch_size, n_sample-cur_i)

            for out_dir in [sample_dir_1, sample_dir_2]:
                os.makedirs(out_dir, exist_ok=True)
                z = torch.randn(cur_batch_size, 512, device=device)
                ims = gnet(z, None)
                ims = ims.permute(0, 2, 3, 1).clamp_(-1, 1).add_(1).div_(2).mul_(255).round_().type(torch.uint8).cpu().numpy()

                for p_i, im in enumerate(ims):
                    # PIL is too slow.
                    # im = Image.fromarray(im, 'RGB')
                    # im.save(f'{out_dir}/{cur_i+p_i}.png')

                    # Use opencv.
                    # im = cv2.imwrite(f'{out_dir}/{cur_i+p_i}.png', im[:, :, ::-1])

                    # Use mutlithread to release io wait.
                    q.put([f'{out_dir}/{cur_i+p_i}.png', im])

    q.put(None)
    q.join()
    print('Complete.')
