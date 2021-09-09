def binary_change(Imm_1,Imm_2):
    outImage=np.empty(Imm_1.shape)
    outImage[:]=np.nan
    for r in range (outImage.shape[0]):
        for c in range(outImage.shape[1]):
            if Imm_1[r][c] == 1 and Imm_2[r][c]== 1:
                outImage[r][c]=1
            elif Imm_1[r][c] == 0 or Imm_2[r][c]== 0:
                outImage[r][c]=0                               
            elif math.isnan(Imm_1[r][c]) or math.isnan(Imm_2[r][c]):
                outImage[r][c]=np.nan
    plt.figure(figsize=(9, 4))
    plt.subplot(111)
    plt.imshow(outImage, cmap='gray', interpolation='nearest')
    plt.axis('off')              
    return outImage 
