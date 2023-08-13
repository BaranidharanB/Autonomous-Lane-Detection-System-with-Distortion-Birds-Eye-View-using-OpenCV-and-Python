import cv2, numpy as np

ImageFeed = cv2.VideoCapture("lanes_clip.mp4")

success, image = ImageFeed.read()

while success:
    success,image = ImageFeed.read()
    Frame = cv2.resize(image,(720,720))

    # Choosing the co-ordinates

    TopLeft = (250,387)
    BotLeft = (210,482)
    TopRight = (400,380)
    BotRight = (460,472)

    cv2.circle(Frame,TopLeft,7,(0,255,0),-1)
    cv2.circle(Frame,BotLeft,7,(0,255,0),-1) 
    cv2.circle(Frame,TopRight,7,(0,255,0),-1)
    cv2.circle(Frame,BotRight,7,(0,255,0),-1)

    # Applying the Geometrical Transformation

    Points1 = np.float32([TopLeft,BotLeft, TopRight,BotRight])
    Points2 = np.float32([[0,0],[0,720], [720,0],[720,720]])

    # This Matrix Returns and it returns 2D array. 
    # Basically it multiplies between the co-ordinates of matrix and gives output transformed co-ordinates
    Matrix = cv2.getPerspectiveTransform(Points1,Points2)
    TransformedImage = cv2.warpPerspective(Frame,Matrix,(720,720))


    hsv = cv2.cvtColor(Frame, cv2.COLOR_BGR2HSV)
    Transformedhsv = cv2.cvtColor(TransformedImage, cv2.COLOR_BGR2HSV)


    # lowerLimit = np.array([0,0,130])
    # upperLimit = np.array([255,50,255])
    lowerLimit = np.array([0,0,150])
    upperLimit = np.array([255,255,255])
    Mask = cv2.inRange(hsv,lowerLimit,upperLimit)
    TransformedMask = cv2.inRange(Transformedhsv,lowerLimit,upperLimit)
    Output = cv2.bitwise_and(Frame,Frame,Mask)

    HistoGram = np.sum(Mask[Mask.shape[0]//2:,:],axis=0)
    MidPoint = int(HistoGram.shape[0]/2) # This is to detect left or right lane 
    LeftBase = np.argmax(HistoGram[:MidPoint])
    RightBase = np.argmax(HistoGram[MidPoint:]) + MidPoint

    # Creating Sliding Window to track the lane
    y = 712
    LeftX = []
    RightX = []

    MaskCopy = Mask.copy()

    while y > 0:
        
        Img = Mask[y-40:y,LeftBase-50:LeftBase+50]
        # This Contours will able to detect the left side of the lane through the window
        Contours, _ = cv2.findContours(Img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) 
        for contour in Contours:
            moments = cv2.moments(contour)
            if moments['m00'] != 0:
                cx = int(moments["m10"]/moments["m00"])
                cy = int(moments["m01"]/moments["m00"])  
                LeftX.append(LeftBase - 50 + cx)
                LeftBase = LeftBase - 50 + cx
        
        Img = Mask[y-40:y,RightBase-50:RightBase+50]
        # This Contours will able to detect the left side of the lane through the window
        Contours, _ = cv2.findContours(Img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) 
        for contour in Contours:
            moments = cv2.moments(contour)
            if moments['m00'] != 0:
                cx = int(moments["m10"]/moments["m00"])
                cy = int(moments["m01"]/moments["m00"])  
                LeftX.append(RightBase - 50 + cx)
                RightBase = RightBase - 50 + cx

        cv2.rectangle(MaskCopy,(LeftBase - 50,y),(LeftBase+50,y-40),(255,255,255),2)
        cv2.rectangle(MaskCopy,(RightBase - 50,y),(RightBase+50,y-40),(255,255,255),2)
        y = y - 40

#-------# 
    HistoGram = np.sum(TransformedMask[TransformedMask.shape[0]//2:,:],axis=0)
    MidPoint = int(HistoGram.shape[0]/2) # This is to detect left or right lane 
    LeftBase = np.argmax(HistoGram[:MidPoint])
    RightBase = np.argmax(HistoGram[MidPoint:]) + MidPoint

    # Creating Sliding Window to track the lane
    y = 712
    LeftX = []
    RightX = []

    TrasformedMaskCopy = TransformedMask.copy()

    while y > 0:
        
        Img = TransformedMask[y-40:y,LeftBase-50:LeftBase+50]
        # This Contours will able to detect the left side of the lane through the window
        Contours, _ = cv2.findContours(Img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) 
        for contour in Contours:
            moments = cv2.moments(contour)
            if moments['m00'] != 0:
                cx = int(moments["m10"]/moments["m00"])
                cy = int(moments["m01"]/moments["m00"])  
                LeftX.append(LeftBase - 50 + cx)
                LeftBase = LeftBase - 50 + cx
        
        Img = TransformedMask[y-40:y,RightBase-50:RightBase+50]
        # This Contours will able to detect the left side of the lane through the window
        Contours, _ = cv2.findContours(Img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) 
        for contour in Contours:
            moments = cv2.moments(contour)
            if moments['m00'] != 0:
                cx = int(moments["m10"]/moments["m00"])
                cy = int(moments["m01"]/moments["m00"])  
                LeftX.append(RightBase - 50 + cx)
                RightBase = RightBase - 50 + cx

        cv2.rectangle(TrasformedMaskCopy,(LeftBase - 50,y),(LeftBase+50,y-40),(255,255,255),2)
        cv2.rectangle(TrasformedMaskCopy,(RightBase - 50,y),(RightBase+50,y-40),(255,255,255),2)
        y = y - 40


    # Birds Eye View Mask
    TransformedImageHSV = cv2.cvtColor(TransformedImage, cv2.COLOR_BGR2HSV)
    TransformedImageMask = cv2.inRange(TransformedImageHSV,lowerLimit,upperLimit)





    # cv2.imshow("HSV Frame",Output)
    # cv2.imshow("Mask Frame",Mask)
    # # cv2.imshow("Bird's Eye Mask View Frame",TransformedImageMask)
    # cv2.imshow("Lane Detection Frame",Frame)
    # # cv2.imshow("Bird's Eye View Frame",TransformedImage)
    # cv2.imshow("Sliding Window",MaskCopy)
    # cv2.imshow("Sliding Window Bird's Eye View",TrasformedMaskCopy)



    


    if cv2.waitKey(10) & 0xFF ==ord('q'):
        Frame.release()
        cv2.destroyAllWindows()