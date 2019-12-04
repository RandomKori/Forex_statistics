import pandas as pd
import numpy as np

def createZigZagPoints(dfSeries, minSegSize=2, sizeInDevs=1):
    minRetrace = minSegSize
    
    curVal = dfSeries[0]
    curPos = dfSeries.index[0]
    curDir = 1
    #dfRes = pd.DataFrame(np.zeros((len(dfSeries.index), 2)), index=dfSeries.index, columns=["Dir", "Value"])
    dfRes = pd.DataFrame(index=dfSeries.index, columns=["Dir", "Value"])
    #print(dfRes)
    #print(len(dfSeries.index))
    for ln in dfSeries.index:
        if((dfSeries[ln] - curVal)*curDir >= 0):
            curVal = dfSeries[ln]
            curPos = ln
            #print(str(ln) + ": moving curVal further, to " + str(curVal))
        else:      
            retracePrc = abs((dfSeries[ln]-curVal)/curVal*100)
            #print(str(ln) + ": estimating retracePrc, it's " + str(retracePrc))
            if(retracePrc >= minRetrace):
                #print(str(ln) + ": registering key point, its pos is " + str(curPos) + ", value = " + str(curVal) + ", dir=" +str(curDir))
                dfRes.ix[curPos, 'Value'] = curVal
                dfRes.ix[curPos, 'Dir'] = curDir
                curVal = dfSeries[ln]
                curPos = ln
                curDir = -1*curDir
                #print(str(ln) + ": setting new cur vals, pos is " + str(curPos) + ", curVal = " + str(curVal) + ", dir=" +str(curDir))
        #print(ln, curVal, curDir)
    dfRes[['Value']] = dfRes[['Value']].astype(float)
    dfRes = dfRes.interpolate(method='linear')
    return dfRes
