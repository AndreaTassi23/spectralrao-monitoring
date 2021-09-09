

def calculateDistance(x1,y1,x2,y2):  
    dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)  
    return dist  


def threshold_method(diffImmage):
    outImaSec=np.empty(diffImmage.shape)
    outImaSec[:]=np.nan
    diffImmageLin=diffImmage.ravel()
    diffImmageLin = diffImmageLin[~np.isnan(diffImmageLin)]
    # Secant section
    density = stats.gaussian_kde(diffImmageLin)
    n,xtemp=np.histogram(diffImmageLin)
    distributionLine=plt.plot(xtemp,density(xtemp))
    x,y=distributionLine[0].get_data()
    maxInd=np.argmax(y)
    xMax=x[maxInd]
    yMax=y[maxInd]
    minInd=np.argmin(y)
    xMin=x[minInd]
    yMin=y[minInd]
    mSec= (yMin-yMax)/(xMin-xMax)
    qSec=(xMin*yMax - xMax*yMin)/(xMin-xMax)
    maxDistance=0
    maxDistanceInd=maxInd
    mPerp=-1/mSec
    for tempInd in range(maxInd,minInd):
        xTemp=x[tempInd]
        yTemp=y[tempInd]
        qTemp= yTemp-mPerp*xTemp
        xInter= (qTemp-qSec) / (mSec-mPerp)
        yInter= mSec*xInter + qSec
        dist=calculateDistance(xTemp,yTemp,xInter,yInter)
        if dist > maxDistance:
            maxDistance=dist
            maxDistanceInd=tempInd
    tSec=x[maxDistanceInd]
    print("Threshold value = ", tSec)
    
    for r in range(diffImmage.shape[0]):
        for c in range(diffImmage.shape[1]):
            if diffImmage[r][c] > tSec:
                outImaSec[r][c]=1
            elif not np.isnan(diffImmage[r][c]):
                outImaSec[r][c]=0
    plt.figure()            
    hist, bins_center = exposure.histogram(diffImmageLin)
    plt.plot(bins_center, hist, lw=2)       
    plt.axvline(tSec, color='k', ls='--')
    
    plt.figure(figsize=(9, 4))
    plt.subplot(131)
    plt.imshow(diffImmage, cmap='gray', interpolation='nearest')
    plt.axis('off')
    
    return diffImmage
