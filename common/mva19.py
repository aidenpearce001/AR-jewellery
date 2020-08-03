'''
Adapted from the MonoHand3D codebase for the MonocularRGB_2D_Handjoints_MVA19 project (github release)

@author: Paschalis Panteleris (padeler@ics.forth.gr)
'''

import numpy as np
import cv2
import tensorflow as tf
from scipy.ndimage.filters import gaussian_filter


# find connection in the specified sequence, center 29 is in the position 15
limbSeq = [[0, 1], [0, 5], [0, 9], [0, 13], [0, 17], # palm
           [1, 2], [2, 3], [3,4], # thump
           [5, 6], [6, 7], [7, 8], # index
           [9, 10], [10, 11], [11, 12], # middle
           [13, 14], [14, 15], [15, 16], # ring
           [17, 18], [18, 19], [19, 20], # pinky
        ]

# visualize
colors = [[255,255,255], 
          [255, 0, 0], [255, 60, 0], [255, 120, 0], [255, 180, 0],
          [0, 255, 0], [60, 255, 0], [120, 255, 0], [180, 255, 0],
          [0, 255, 0], [0, 255, 60], [0, 255, 120], [0, 255, 180],
          [0, 0, 255], [0, 60, 255], [0, 120, 255], [0, 180, 255],
          [0, 0, 255], [60, 0, 255], [120, 0, 255], [180, 0, 255],]


def visualize_2dhand(canvas, hand, scale, stickwidth = 3, thre=0.1):
    radius = 20
    for pair in limbSeq:
        if hand[pair[0]][2]>thre and hand[pair[1]][2]>thre:
            x0,y0 = hand[pair[0]][:2]
            x0, y0 = x0*scale[0], y0*scale[1]
            x1,y1 = hand[pair[1]][:2]
            x1, y1 = x1*scale[0], y1*scale[1]
            x0 = int(x0)
            x1 = int(x1)
            y0 = int(y0)
            y1 = int(y1)    
            cv2.circle(canvas, (x0, y0), radius, colors[pair[0]], thickness = -1)
            cv2.circle(canvas, (x1, y1), radius, colors[pair[1]], thickness = -1)
            cv2.line(canvas,(x0,y0), (x1,y1), colors[pair[1]%len(colors)], thickness=stickwidth,lineType=cv2.LINE_AA)
    return canvas

def get_peak(heatmap_avg, thresh = 0.1):
    all_peaks = []
    for part in range(21):
        hmap_ori = heatmap_avg[:, :, part]
        hmap = gaussian_filter(hmap_ori, sigma=3)
        # Find the pixel that has maximum value compared to those around it
        hmap_left = np.zeros(hmap.shape)
        hmap_left[1:, :] = hmap[:-1, :]
        hmap_right = np.zeros(hmap.shape)
        hmap_right[:-1, :] = hmap[1:, :]
        hmap_up = np.zeros(hmap.shape)
        hmap_up[:, 1:] = hmap[:, :-1]
        hmap_down = np.zeros(hmap.shape)
        hmap_down[:, :-1] = hmap[:, 1:]

        # reduce needed because there are > 2 arguments
        peaks_binary = np.logical_and.reduce(
            (hmap >= hmap_left, hmap >= hmap_right, hmap >= hmap_up, hmap >= hmap_down, hmap > thresh))
        peaks = list(zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0]))  # note reverse
        peaks_with_score = [x + (hmap_ori[x[1], x[0]],) for x in peaks]  # add a third element to tuple with score

        all_peaks.append(peaks_with_score)
    return all_peaks

def peaks_to_hand(peaks, dx=0,dy=0):
    hand = []
    for joints in peaks:
        sel = sorted(joints, key=lambda x: x[2], reverse=True)
        
        if len(sel)>0:
            p = sel[0]
            x,y,score = p[0]+dx, p[1]+dy, p[2]
            hand.append([x,y,score])
        else:
            hand.append([0,0,0])
        
    return np.array(hand,dtype=np.float32)


def load_graph(model_file):
    graph = tf.Graph()
    graph_def = tf.compat.v1.GraphDef()

    with open(model_file, "rb") as f:
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.import_graph_def(graph_def)

    return graph

def padRightDownCorner(img, stride, padValue):
    h = img.shape[0]
    w = img.shape[1]

    pad = 4 * [None]
    pad[0] = 0 # up
    pad[1] = 0 # left
    pad[2] = 0 if (h%stride==0) else stride - (h % stride) # down
    pad[3] = 0 if (w%stride==0) else stride - (w % stride) # right

    img_padded = img
    pad_up = np.tile(img_padded[0:1,:,:]*0 + padValue, (pad[0], 1, 1))
    img_padded = np.concatenate((pad_up, img_padded), axis=0)
    pad_left = np.tile(img_padded[:,0:1,:]*0 + padValue, (1, pad[1], 1))
    img_padded = np.concatenate((pad_left, img_padded), axis=1)
    pad_down = np.tile(img_padded[-2:-1,:,:]*0 + padValue, (pad[2], 1, 1))
    img_padded = np.concatenate((img_padded, pad_down), axis=0)
    pad_right = np.tile(img_padded[:,-2:-1,:]*0 + padValue, (1, pad[3], 1))
    img_padded = np.concatenate((img_padded, pad_right), axis=1)

    return img_padded, pad

def preprocess(oriImg, boxsize=368, stride=8, padValue=128):
    scale = float(boxsize) / float(oriImg.shape[0])

    imageToTest = cv2.resize(oriImg, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    imageToTest_padded, pad = padRightDownCorner(imageToTest, stride, padValue)
    input_img = np.transpose(np.float32(imageToTest_padded[:,:,:,np.newaxis]), (3,0,1,2)) # required shape (1, width, height, channels)

    return input_img, pad


def update_bbox(p2d, dims, pad=0.3):
    x = np.min(p2d[:,0])
    y = np.min(p2d[:,1])
    xm = np.max(p2d[:,0])
    ym = np.max(p2d[:,1])

    cx = (x+xm)/2
    cy = (y+ym)/2
    w = xm - x
    h = ym - y
    b = max((w,h,224))
    b = int(b + b*pad)

    x = cx-b/2
    y = cy-b/2

    x = max(0,int(x))
    y = max(0,int(y))

    x = min(x, dims[0]-b)
    y = min(y, dims[1]-b)
    

    return [x,y,b,b]



class Estimator(object):
    def __init__(self, model_file, input_layer="input_1", output_layer="k2tfout_0"):
        
        input_name = "import/" + input_layer
        output_name = "import/" + output_layer

        self.graph = load_graph(model_file)
        for op in self.graph.get_operations():
            print(op.name)
        self.input_operation = self.graph.get_operation_by_name(input_name)
        self.output_operation = self.graph.get_operation_by_name(output_name)
        self.sess = tf.compat.v1.Session(graph=self.graph)
        

    def predict(self, img):
        results = self.sess.run(self.output_operation.outputs[0], feed_dict={self.input_operation.outputs[0]: img})
        return np.squeeze(results)

