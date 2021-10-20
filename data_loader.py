import pandas as pd
import numpy as np

def loadData(fileName):
    df = pd.read_csv(fileName)
    labels = df.iloc[:, 0]
    images = df.iloc[:, 1:785]
    np_labels = labels.to_numpy()
    np_images = images.to_numpy()
    #normalize to 0-1.0
    np_images_processed = np.reshape(np_images, (-1, 28, 28, 1))/255.0
    return np_labels, np_images_processed

