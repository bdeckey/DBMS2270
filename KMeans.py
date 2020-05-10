import plotly.graph_objects as go
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from sklearn.cluster import KMeans
import csv
import pywt
import math

def sliding_chunker(data, window_len, slide_len):
    """
    Split a list into a series of sub-lists, each sub-list window_len long,
    sliding along by slide_len each time. If the list doesn't have enough
    elements for the final sub-list to be window_len long, the remaining data
    will be dropped.
    e.g. sliding_chunker(range(6), window_len=3, slide_len=2)
    gives [ [0, 1, 2], [2, 3, 4] ]
    """
    chunks = []
    for pos in range(0, len(data), slide_len):
        chunk = np.copy(data[pos:pos+window_len])
        if len(chunk) != window_len:
            continue
        chunks.append(chunk)

    return chunks

def plot_waves(waves, step):
    """
    Plot a set of 9 waves from the given set, starting from the first one
    and increasing in index by 'step' for each subsequent graph
    """
    plt.figure()
    n_graph_rows = 3
    n_graph_cols = 3
    graph_n = 1
    wave_n = 0
    for _ in range(n_graph_rows):
        for _ in range(n_graph_cols):
            axes = plt.subplot(n_graph_rows, n_graph_cols, graph_n)
            axes.set_ylim([-100, 150])
            plt.plot(waves[wave_n])
            graph_n += 1
            wave_n += step
    # fix subplot sizes so that everything fits
    plt.tight_layout()
    plt.show()

def reconstruct(data, window, clusterer):
    """
    Reconstruct the given data using the cluster centers from the given
    clusterer.
    """
    window_len = len(window)
    slide_len = int(window_len/2)
    segments = sliding_chunker(data, window_len, slide_len)
    reconstructed_data = np.zeros(len(data))
    for segment_n, segment in enumerate(segments):
        # window the segment so that we can find it in our clusters which were
        # formed from windowed data
        segment *= window
        nearest_match_idx = clusterer.predict(segment.reshape(1, -1))[0]
        nearest_match = np.copy(clusterer.cluster_centers_[nearest_match_idx])

        pos = segment_n * slide_len
        reconstructed_data[pos:pos+window_len] += nearest_match

    return reconstructed_data

def get_windowed_segments(data, window):
    """
    Populate a list of all segments seen in the input data.  Apply a window to
    each segment so that they can be added together even if slightly
    overlapping, enabling later reconstruction.
    """
    step = 2
    windowed_segments = []
    segments = sliding_chunker(
        data,
        window_len=len(window),
        slide_len=step
    )
    for segment in segments:
        segment *= window
        windowed_segments.append(segment)
    return windowed_segments

def find_nearest(array, value, array2):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    if idx == 0:
        return array[idx], array[idx + 1], array2[idx], array2[idx + 1]
    if idx == len(array)-1:
        return array[idx-1], array[idx], array2[idx-1], array2[idx]
    if value > array[idx]:
        return array[idx], array[idx+1], array2[idx], array2[idx + 1]
    else:
        return array[idx-1], array[idx], array2[idx-1], array2[idx]

def findTotal(x, y, sampleX, sampleY):
    maxdev = []
    tot = 0
    curMax = [0, 0]
    sx1, sx2 = 0, 0
    for i in range(len(x)):
        if i > 0 and i > sx1 and i < sx2:
            line = m * x[i] + b
            dist = np.abs(line - y[i])
            if dist > curMax[0]:
                curMax = [dist, i]
            maxdev += [dist]
            tot += dist
        else:
            sx1, sx2, sy1, sy2 = find_nearest(sampleX, x[i], sampleY)

            if (sx1 == sx2):
                print('equality error')
            m = (sy1 - sy2) / (sx1 - sx2)
            b = (sx1 * sy2 - sx2 * sy1) / (sx1 - sx2)
            line = m * x[i] + b
            dist = np.abs(line - y[i])
            if dist > curMax[0]:
                curMax = [dist, i]
            maxdev += [dist]
            tot += dist
    return tot

def main():
    """
    Main function.
    """
    df = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/finance-charts-apple.csv")

    del df['AAPL.Open']
    del df['direction']
    del df['up']
    del df['AAPL.Low']
    del df['AAPL.Close']
    del df['AAPL.Volume']
    del df['AAPL.Adjusted']
    del df['dn']
    del df['mavg']
    del df['Date']

    d = df.reset_index().values
    xcol = d[:, 0]
    data = d[:, 1]

    y = []
    x = []
    WINDOW_LEN = 32


    # with open('ECGTime.csv', newline='') as csvfile:
    #     reader = csv.reader(csvfile, delimiter=',')
    #     for row in reader:
    #         x += [int(row[0])]
    #         y += [float(row[1])]
    # print("Reading data...")
    # data = np.array(y[:500])
    # data = y[:500]

    window_rads = np.linspace(0, np.pi, WINDOW_LEN)
    window = np.sin(window_rads)**2
    print("Windowing data...")
    windowed_segments = get_windowed_segments(data, window)

    print("Clustering...")
    clusterer = KMeans(n_clusters=150)
    clusterer.fit(windowed_segments)

    print("Reconstructing...")
    reconstruction = reconstruct(data, window, clusterer)
    error = reconstruction - data
    print("Maximum reconstruction error is %.1f" % max(error))
    # print(reconstruction[0:10])
    print('error', findTotal(xcol[20:480], data[20:480], xcol[20:480], reconstruction[20:480]))
    print(reconstruction[:5])
    plt.figure()
    plt.plot(data, label="Original Stock Price")
    plt.plot(reconstruction, label="K-Means Reconstructtion")
    # plt.plot(error, label="Reconstruction error")
    plt.legend()
    plt.show()

main()

