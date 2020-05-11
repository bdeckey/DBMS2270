from scipy.spatial import distance
from pyts.approximation import PiecewiseAggregateApproximation
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from sklearn.cluster import KMeans
import csv
import pywt
import math

def getStockData():
    """Get data from a local csv and remove the columns that are not needed. Returns
    2 arrays containing information"""

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

    data = df.reset_index().values
    xcol = data[:, 0]
    ycol = data[:, 1]

    return xcol, ycol

def find_nearest(array, value, array2):
    """Used in the creation of APCA, this function returns the next valid
    index when trying to match up a shortened APCA representation into a valid
    array for graphing, (IE APCA needs to have the same number of samples when graphed as the
    original data"""
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

def findMax(x, y, sampleX, sampleY):
    """This function finds the point on a representation that is the farthest away from its counterpart on the
    real TS"""
    maxdev = []
    curMax = [0, 0]
    sx1, sx2 = 0, 0
    for i in range(len(x)):
        if i > 0 and i > sx1 and i < sx2:
            line = m * x[i] + b
            dist = np.abs(line - y[i])
            if dist > curMax[0]:
                curMax = [dist, i]
            maxdev += [dist]
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
    return curMax

def findTotal(x, y, sampleX, sampleY):
    """ This function was used before we used L-norms to find the total euclidean distance difference
    between the sample and the real TS"""
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

def APCA(x, y, sampleX):
    """ This function returns the APCA representation and the array that can be graphed to view the
    representation given a series of segments and the real data"""
    APCArep = []
    APCAgraphY = []
    curInd = 1
    curSum = 0
    counter = 0
    for i in range(len(x)):
        if i < np.floor(sampleX[curInd]):
            counter += 1
            curSum += y[i]
        else:
            curInd += 1
            mean = curSum/counter
            APCArep += [(mean, i)]
            counter = 1
            curSum = y[i]
    curInd = 0
    for i in range(len(x)):
        if i <= APCArep[curInd][1]:
            APCAgraphY += [APCArep[curInd][0]]
        else:
            curInd += 1
            APCAgraphY += [APCArep[curInd][0]]
    return APCArep, APCAgraphY



def curveFitting(x, y, errorTol, numseg):
    """This function is the main function for APCA as it finds the
    break points and where to put the next segment in regards to APCA"""
    length = len(x)
    sampleX = np.linspace(x[0], x[len(x)-1], int(numseg))
    sampleY = np.interp(sampleX, x, y)
    maxDev = findMax(x,y, sampleX, sampleY)

    if maxDev[1]+1 == length:
        # print('call off too dangerous!!')
        return sampleX, sampleY
    if maxDev[1] == 0:
        # print('call off too dangerous!!')
        return sampleX, sampleY
    if len(sampleX) < 2:
        # print('call off too dangerous!!')
        return sampleX, sampleY

    if maxDev[0] > errorTol:
        segi = np.ceil(numseg * (maxDev[1]/length))
        segii = numseg - segi

        if segii == 0 or segii == 1:
            segii = 2
        if segi == 0 or segi == 1:
            segi = 2

        xi = x[:maxDev[1]]
        yi = y[:maxDev[1]]
        sxi = np.linspace(x[0], maxDev[1]-1, int(segi))
        syi = np.interp(sxi, xi, yi)

        xii = x[maxDev[1]+1:]
        yii = y[maxDev[1]+1:]
        sxii = np.linspace(maxDev[1]+1, x[len(x)-1], int(segii))
        syii = np.interp(sxii, xii, yii)

        curPoint = (maxDev[1], y[maxDev[1]])
        pointi = (sxi[-1], syi[-1])
        pointii = (sxii[0], syii[0])

        dist1 = distance.euclidean(curPoint,  pointi)
        dist2 = distance.euclidean(curPoint, pointii)

        if dist1 >= dist2:
            xii = np.insert(xii, 0, [x[maxDev[1]]])
            yii = np.insert(yii, 0, y[maxDev[1]])
        else:
            xi = np.append(xi, [x[maxDev[1]]])
            yi = np.append(yi, [y[maxDev[1]]])


        finalxi, finalyi = curveFitting(xi, yi, errorTol, segi)
        finalxii, finalyii = curveFitting(xii, yii, errorTol, segii)

        finalx = np.append(finalxi, finalxii)
        finaly = np.append(finalyi, finalyii)
        return finalx, finaly
    else:
        return sampleX, sampleY

def prepend(a1, a2):
    """Simple function to prepend an array to another array"""
    a2.extend(a1)
    a1 = a2
    return a1

def threshold(array, M):
    """This is for wavelet creation of APCA, it was not used in the end.
    This function returns the highest absolute value numbers in the array given and fills the other indices with
    zeros, depending on how high M is"""
    curMinMax = 0
    maxes = []
    for i in range(len(array)):
        curVal = abs(array[i])
        if curVal > curMinMax:
            if len(maxes) < M:
                maxes += [[curVal,i]]
            else:
                temp = min(maxes)
                maxes.remove(temp)
                maxes += [[curVal,i]]
                temp = min(maxes)
                curMinMax = temp[0]
    final = []
    indexes = []
    for i in range(len(maxes)):
        indexes += [maxes[i][1]]
    for i in range(len(array)):
        if i in indexes:
            final += [array[i]]
        else:
            final += [0]
    return final


def reconstruct(coeffs):
    """Another function for Wavelet creation, not used in final results"""
    reconst = []
    reconst.append(coeffs[0]+coeffs[1])
    reconst.append(coeffs[0]-coeffs[1])
    del coeffs[:2]
    while len(reconst) <= len(coeffs):
        length = len(reconst)
        temp = []
        for i in range(length):
            temp += [reconst[i] + coeffs[i]]
            temp += [reconst[i] - coeffs[i]]
        del coeffs[:length]
        reconst = temp
    return reconst


def DWT(array, M):
    """ Another wavelet function not used in final."""
    length = len(array)
    topRes = math.log(length, 2)-1
    if not math.log(length, 2).is_integer():
        print("Length needs to be a power of 2")
        return []
    details = []
    newArray = []
    CurDetails = []
    for i in range(1, length, 2):
        cur = (array[i-1] + array[i])/2
        newArray += [cur]
        CurDetails += [(cur - array[i])/2**(topRes/2)]
    topRes -= 1
    details = prepend(details, CurDetails)
    CurDetails = []
    while len(newArray) != 1:
        temp = []
        for i in range(1, len(newArray), 2):
            cur = (newArray[i-1] + newArray[i])/2
            temp += [cur]
            CurDetails += [(cur - newArray[i])/2**(topRes/2)]
        topRes -= 1
        newArray = temp
        details = prepend(details, CurDetails)
        CurDetails = []
    Coeffs = prepend(details, newArray)
    keep = threshold(Coeffs, M)
    # print('keep', keep)
    answer = reconstruct(keep)
    # print('final', answer)
    return answer

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


def getAPCA(x, y, errorTol, numseg):
    """Just a function to call the appropriate functions for an APCA representation"""
    ans = curveFitting(x, y, errorTol, numseg)
    a = APCA(x, y, ans[0])
    return a[1]

def getEKGData():
    """Getter function for the EKG data"""
    x = []
    y1 = []
    y2 = []
    micro = []

    with open('ECGTime.csv', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            x += [int(row[0])]
            y1 += [float(row[1])]
            y2 += [float(row[2])]
            micro += [[int(row[0]), float(row[1])]]
    return np.array(x[:506]), np.array(y1[:506])


def getPAA(y, window_length):
    """Getter function for the PAA representation"""
    transformer = PiecewiseAggregateApproximation(window_size=window_length)
    paa = transformer.transform(y.reshape(1, -1))
    paaX = []
    linpaa = []
    paa_graph = []
    paa_graphX = []
    for i in range(len(paa[0])):
        if i == 0:
            paa_graph += [paa[0][i]]
            paa_graph += [paa[0][i + 1]]
            curX = i * window_length
            paa_graphX += [curX]
            paa_graphX += [curX + window_length]
        elif i != len(paa[0]) - 1:
            paa_graph += [paa[0][i + 1]]
            paa_graph += [paa[0][i + 1]]
            curX = i * window_length
            paa_graphX += [curX]
            paa_graphX += [curX + window_length]
        paaX += [i * window_length]
    for i in range(len(paa[0])):
        for j in range(window_length):
            linpaa += [paa[0][i]]
    return paa_graphX, paa_graph, linpaa[:len(y)]


def getKMeans(y, window_length, clusters):
    """Calls the necesary functions for KMeans given a window length and number of clusters"""
    window_rads = np.linspace(0, np.pi, window_length)
    window = np.sin(window_rads) ** 2
    windowed_segments = get_windowed_segments(y, window)
    clusterer = KMeans(n_clusters=clusters)
    clusterer.fit(windowed_segments)
    reconstruction = reconstruct(y, window, clusterer)
    return reconstruction

def lnorm(real, aprox):
    """ Calls lnorm for multiple p values"""
    dif = real-aprox
    l1 = np.linalg.norm(dif, 1)
    l2 = np.linalg.norm(dif, 2)
    linf = np.linalg.norm(dif, np.inf)
    return l1,l2,linf

def findBestAPCA(data):
    """Function testing different values for paramters in APCA"""
    error = 0.3
    length = [30, 45, 50, 55, 60, 70, 80, 90]
    max1 = 1000000000
    max1ind = 0
    max2 = 1000000000
    max2ind = 0
    max3 = 1000000000
    max3ind = 0
    for i in range(len(error)):
        for j in range(len(length)):
            e = error[i]
            l = length[j]
            apcaData = getAPCA(data[0], data[1], e, l)
            norms = lnorm(data[1][15:480], apcaData[15:480])
            if max1 > norms[0]:
                max1 = norms[0]
                max1ind = [e, l]
            if max2 > norms[1]:
                max2 = norms[1]
                max2ind = [e, l]
            if max3 > norms[2]:
                max3 = norms[2]
                max3ind = [e, l]
    print('l1', max1, max1ind)
    print('l2', max2, max2ind)
    print('linfinity', max3, max3ind)
    return max2ind

def findBestKmeans(data):
    """Function testing different Parameters for K-means"""
    # window = 32
    # clusters = 150
    # stockKMeans = getKMeans(stockData[1], window, clusters)
    windows = [32,16,48,10,6,15]
    clusters = [150,100,175,50]
    max1 = 1000000000
    max1ind = 0
    max2 = 1000000000
    max2ind = 0
    max3 = 1000000000
    max3ind = 0
    for i in range(len(windows)):
        for j in range(len(clusters)):
            w = windows[i]
            c = clusters[j]
            stockKMeans = getKMeans(data[1], w, c)
            norms = lnorm(data[1][15:480], stockKMeans[15:480])
            if max1 > norms[0]:
                max1 = norms[0]
                max1ind = [w, c]
            if max2 > norms[1]:
                max2 = norms[1]
                max2ind = [w, c]
            if max3 > norms[2]:
                max3 = norms[2]
                max3ind = [w, c]
    print('l1', max1, max1ind)
    print('l2', max2, max2ind)
    print('linfinity', max3, max3ind)
    return max2ind
