from functions import *

if __name__ == '__main__':
    stockData = getStockData()
    EKGData = getEKGData()

    ## STOCK MARKET DATA ##
    print("STOCK MARKET DATA | l1 | l2 | l-infinity")
    ## APCA
    errorTol = 8
    length = 50
    stockAPCA = getAPCA(stockData[0], stockData[1], errorTol, length)
    print("APCA", lnorm(stockData[1][15:480], stockAPCA[15:480]))

    ## PAA
    window = 5
    stockPAA = getPAA(stockData[1], window)
    print("PAA", lnorm(stockData[1][15:480], stockPAA[2][15:480]))

    ## K-Means 15 - 480
    window = 32
    clusters = 150
    stockKMeans = getKMeans(stockData[1], window, clusters)
    print("K-Means", lnorm(stockData[1][15:480], stockKMeans[15:480]))


    ## EKG DATA ##
    print("EKG DATA | l1 | l2 | l-infinity")
    ## APCA
    errorTol = 0.5
    length = 60
    EKGapca = getAPCA(EKGData[0], EKGData[1], errorTol, length)
    print("APCA", lnorm(EKGData[1][15:480], EKGapca[15:480]))

    ## PAA
    window = 5
    EKGPAA = getPAA(EKGData[1], window)
    print("PAA", lnorm(EKGData[1][15:480], EKGPAA[2][15:480]))

    ## K-Means 15 - 480
    window = 32
    clusters = 150
    EKGKMeans = getKMeans(EKGData[1], window, clusters)
    print("K-Means", lnorm(EKGData[1][15:480], EKGKMeans[15:480]))

    # plt.plot(stockData[1], label="Stock Market Price")
    # plt.plot(stockAPCA, label="APCA")
    # plt.plot(stockPAA[2], label="PAA")
    # plt.plot(stockKMeans, label="K-Means")
    # plt.legend()
    # plt.show()

    # plt.plot(EKGData[1], label="EKG Price")
    # plt.plot(EKGapca, label="APCA")
    # plt.plot(EKGPAA[2], label="PAA")
    # plt.plot(EKGKMeans, label="K-Means")
    # plt.legend()
    # plt.show()


