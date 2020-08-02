import cv2
import time
import numpy as np

protoFile = "pose_deploy.prototxt"
weightsFile = "pose_iter_102000.caffemodel"
nPoints = 22
POSE_PAIRS = [ [0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[0,9],[9,10],[10,11],[11,12],[0,13],[13,14],[14,15],[15,16],[0,17],[17,18],[18,19],[19,20] ]
net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

cap = cv2.VideoCapture("hand.mp4")
# cap = cv2.VideoCapture("hand_test.mp4")
# out = cv2.VideoWriter('output.mp4', -1, 20.0, (640,480))

# print('Video Dimensions: ',get_vid_properties())
if (cap.isOpened()== False): 
    print("Error opening video stream or file")
print(cap.isOpened())
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:

#     cv2.imshow('Frame',frame)

        frameCopy = np.copy(frame)
        frameWidth = frame.shape[1]
        frameHeight = frame.shape[0]
        print(frameWidth,frameHeight)
        aspect_ratio = frameWidth/frameHeight
        threshold = 0.1

        t = time.time()
        # input image dimensions for the network
        inHeight = 368
        inWidth = int(((aspect_ratio*inHeight)*8)//8)
        inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight),
                                  (0, 0, 0), swapRB=False, crop=False)
        
        net.setInput(inpBlob)

        points = []
        output = net.forward()
        for i in [13, 14]:
    # confidence map of corresponding body's part.
            probMap = output[0, i, :, :]
            probMap = cv2.resize(probMap, (frameWidth, frameHeight))
        #     print(cv2.minMaxLoc(probMap))
            # Find global maxima of the probMap.
            minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

            if prob > threshold :
        #         cv2.circle(frameCopy, (int(point[0]), int(point[1])), 8, (0, 255, 255), thickness=4, lineType=cv2.FILLED)
        #         cv2.putText(frameCopy, "{}".format(i), (int(point[0]), int(point[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, lineType=cv2.LINE_AA)

                # Add the point to the list if the probability is greater than the threshold
                points.append((int(point[0]), int(point[1])))
        #         plt.figure(figsize=[14,10])
        #         plt.imshow(frameCopy)

            else :
                points.append(None)
        print(points)
        cv2.circle(frameCopy, (int((points[0][0]+points[1][0])/2), int((points[1][1]+points[0][1])/2)), 8, (0, 255, 255), thickness=4, lineType=cv2.FILLED)        
        cv2.imshow('Frame',frameCopy)
#         cap.set(cv2.cv.CV_CAP_PROP_POS_AVI_RATIO, 0)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break


cap.release()
# out.release()
cv2.destroyAllWindows()