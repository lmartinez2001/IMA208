import numpy as np

def medianFilter(origine, kernel_shape=10):
    median = np.zeros(origine.shape)

    padding_dim = int((kernel_shape - 1) / 2)

    padded = np.pad(origine, padding_dim, mode='constant')

    for i in range(origine.shape[1]):
        for j in range(origine.shape[0]):
            median[j,i] = np.median(padded[j:j+kernel_shape, i:i+kernel_shape])

    return median