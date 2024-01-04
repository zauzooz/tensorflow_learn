from tqdm import tqdm_gui
from time import sleep
import numpy as np

def _Shurima_shuffle():
    print("FAKER!!!!!")

if __name__=="__main__":
    epochs = 20

    batchs = np.random.uniform(0, 1, size=(6, 10, 30)).tolist()
    y1 = np.random.randint(0, 2, size=(6)).tolist()
    y2 = np.random.randint(0, 2, size=(6)).tolist()
    y3 = np.random.randint(0, 2, size=(6)).tolist()

    for epoch in range(1, epochs+1):
        _Shurima_shuffle()
        with tqdm_gui(zip(*[batchs, y1, y2, y3]), unit='batch') as tepoch:
            for data, label1, label2, label3 in tepoch:
                tepoch.set_description(f"Epoch {epoch}")
                # training step here
                loss = np.random.uniform(0,1,size=1)
                accuracy = np.random.uniform(0,1,size=1)
                tepoch.set_postfix(loss=loss.item(), accuracy=100.*accuracy)
                sleep(0.1)