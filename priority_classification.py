def priority_classification(RaoClass,NDVIClass):
    # 9 possible classes combination: 
    outSimplified= np.zeros(RaoClass.shape)
    for r in range (outSimplified.shape[0]):
        for c in range(outSimplified.shape[1]):
            if RaoClass[r][c] == 0 and NDVIClass[r][c]== 0:
                outSimplified[r][c]=0
            elif RaoClass[r][c] == 0 and NDVIClass[r][c]== 1:
                outSimplified[r][c]=1 
            elif RaoClass[r][c] == 0 and NDVIClass[r][c]== 2:
                outSimplified[r][c]=2
            elif RaoClass[r][c] == 1 and NDVIClass[r][c]== 0:
                outSimplified[r][c]=2 
            elif RaoClass[r][c] == 1 and NDVIClass[r][c]== 1:
                outSimplified[r][c]=2
            elif RaoClass[r][c] == 1 and NDVIClass[r][c]== 2:
                outSimplified[r][c]=2
            elif math.isnan(RaoClass[r][c]) or math.isnan(NDVIClass [r][c]):
                outSimplified[r][c]=np.nan
    return outSimplified
