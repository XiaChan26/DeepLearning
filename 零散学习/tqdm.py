import time
from tqdm import tqdm

with tqdm(total=200) as pbar:
    for i in range(20):
        pbar.update(10)
        time.sleep(.1)