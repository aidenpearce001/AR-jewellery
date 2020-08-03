

from __future__ import print_function
from common.mva19 import Estimator, preprocess, get_peak, visualize_2dhand, peaks_to_hand
import numpy as np
import cv2
import time
import imutils


if __name__ == "__main__":

    model_file = "./models/mobnet4f_cmu_adadelta_t1_model.pb"
    input_layer = "input_1"
    output_layer = "k2tfout_0"
    video_path = "aidenhand.mp4"
    stride = 4
    boxsize = 224

    estimator = Estimator(model_file, input_layer, output_layer)
    isVideo = False
    # start webcam
    if isVideo:
        cap = cv2.VideoCapture(video_path)
    else:
        cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_BRIGHTNESS, 0.4)

    paused = False
    delay = {False: 1, True: 0}


    k = 0
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = None
    fps = 15
    (h, w) = (None, None)

    while k != ord('q'):
        ret, frame = cap.read()
        if not ret:
            raise Exception("VideoCapture.read() returned False")
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        if writer is None:
            (h, w) = frame.shape[:2]
            writer = cv2.VideoWriter("testing.avi", fourcc, fps, (w, h), True)
        crop_res = cv2.resize(frame, (boxsize, boxsize))
        img, pad = preprocess(crop_res, boxsize, stride)
        
        tic = time.time()
        hm = estimator.predict(img)
        # print(hm)
        
        dt = time.time() - tic
        print("TTP %.5f, FPS %f" % (dt, 1.0/dt), "HM.shape ", hm.shape)
        hm = cv2.resize(hm, (0, 0), fx=stride, fy=stride)
        #get_scale to convert
        scale_y = h/hm.shape[0]
        scale_x = w/hm.shape[1] 



        peaks = get_peak(hm)
        # print(peaks)
        hand_kp = peaks_to_hand(peaks)
        # print(hand_kp)
        frame = visualize_2dhand(frame, hand_kp, (scale_x, scale_y))
        bg = cv2.normalize(hm[:, :, -1], None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
        # for idx in range(hm.shape[-1]):
        viz = cv2.normalize(np.sum(hm[:, :, :-1], axis=2), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
        # hm = cv2.resize(hm, (frame.shape[1], frame.shape[0]))
        # cv2.imshow("Background", hm[:, :, 0])
        # cv2.imshow("All joint heatmaps", viz)
        cv2.imshow("Input frame", frame)
        writer.write(frame)

        k = cv2.waitKey(delay[paused])

        if k & 0xFF == ord('p'):
            paused = not paused
