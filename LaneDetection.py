import cv2, numpy as np

ImageFeed = cv2.VideoCapture("lanes_clip.mp4")

success, image = ImageFeed.read()

while success:
    success,image = ImageFeed.read()
    Frame = cv2.resize(image,(720,720))

    # Choosing the co-ordinates

    TopLeft = (180,387)
    BotLeft = (70,472)
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
    # lowerLimit = np.array([0,0,168])
    # upperLimit = np.array([255,50,255])
    lowerLimit = np.array([0,0,130])
    upperLimit = np.array([255,50,255])
    Mask = cv2.inRange(hsv,lowerLimit,upperLimit)
    Output = cv2.bitwise_and(Frame,Frame,Mask)

    # Birds Eye View Mask
    TransformedImageHSV = cv2.cvtColor(TransformedImage, cv2.COLOR_BGR2HSV)
    TransformedImageMask = cv2.inRange(TransformedImageHSV,lowerLimit,upperLimit)


    





    cv2.imshow("HSV Frame",Output)
    cv2.imshow("Mask Frame",Mask)
    cv2.imshow("Bird's Eye Mask View Frame",TransformedImageMask)
    cv2.imshow("Lane Detection Frame",Frame)
    cv2.imshow("Bird's Eye View Frame",TransformedImage)


    


    if cv2.waitKey(10) & 0xFF ==ord('q'):
        Frame.release()
        cv2.destroyAllWindows()