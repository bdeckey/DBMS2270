from functions import *

if __name__ == '__main__':
    stockData = getStockData()
    EKGData = getEKGData()

    ## STOCK MARKET DATA ##
    print("STOCK MARKET DATA | l1 | l2 | l-infinity")
    # APCA
    errorTol = 7
    length = 80
    techniques = ['APCA', 'PAA', 'KMEANS']
    stockAPCA = getAPCA(stockData[0], stockData[1], errorTol, length)
    print("APCA", lnorm(stockData[1][15:480], stockAPCA[15:480]))
    APCA_L = lnorm(stockData[1][15:480], stockAPCA[15:480])

    ## PAA
    window = 4
    stockPAA = getPAA(stockData[1], window)
    print("PAA", lnorm(stockData[1][15:480], stockPAA[2][15:480]))
    PAA_L = lnorm(stockData[1][15:480], stockPAA[2][15:480])

    ## K-Means 15 - 480
    window = 15
    clusters = 175
    stockKMeans = getKMeans(stockData[1], window, clusters)
    print("K-Means", lnorm(stockData[1][15:480], stockKMeans[15:480]))
    KMeans_L = lnorm(stockData[1][15:480], stockKMeans[15:480])
    L_1 = [APCA_L[0],PAA_L[0], KMeans_L[0]]
    L_2= [APCA_L[1],PAA_L[1], KMeans_L[1]]
    L_inf = [APCA_L[2],PAA_L[2], KMeans_L[2]]
    plt.bar(techniques, L_inf)
    plt.title('L_inf norm')
    plt.show()
    ## EKG DATA ##
    print("EKG DATA | l1 | l2 | l-infinity")
    ## APCA
    errorTol = 0.3
    length = 55
    EKGapca = getAPCA(EKGData[0], EKGData[1], errorTol, length)
    print("APCA", lnorm(EKGData[1][15:480], EKGapca[15:480]))
    EKGAPCA_L = lnorm(EKGData[1][15:480], EKGapca[15:480])
    ## PAA
    window = 4
    EKGPAA = getPAA(EKGData[1], window)
    print("PAA", lnorm(EKGData[1][15:480], EKGPAA[2][15:480]))
    EKGPAA_L = lnorm(EKGData[1][15:480], EKGPAA[2][15:480])
    ## K-Means 15 - 480
    window = 48
    clusters = 175
    EKGKMeans = getKMeans(EKGData[1], window, clusters)
    print("K-Means", lnorm(EKGData[1][15:480], EKGKMeans[15:480]))
    EKGKMeans_L = lnorm(EKGData[1][15:480], EKGKMeans[15:480])
    l_1 = [EKGAPCA_L[0], EKGPAA_L[0], EKGKMeans_L[0]]
    l_2 = [EKGAPCA_L[1], EKGPAA_L[1], EKGKMeans_L[1]]
    l_inf = [EKGAPCA_L[2], EKGPAA_L[2], EKGKMeans_L[2]]

    # plt.plot(stockData[1][15:480], label="Stock Market Price")
    # plt.plot(stockAPCA[15:480], label="APCA")
    # plt.plot(stockPAA[2][15:480], label="PAA")
    # plt.plot(stockKMeans[15:480], label="K-Means")
    # plt.legend()
    # plt.show()

    # plt.plot(EKGData[1][15:480], label="EKG Price")
    # plt.plot(EKGapca[15:480], label="APCA")
    # plt.plot(EKGPAA[2][15:480], label="PAA")
    # plt.plot(EKGKMeans[15:480], label="K-Means")
    # plt.legend()
    # plt.show()


