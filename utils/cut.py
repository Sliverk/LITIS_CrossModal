import cv2
import gc


def cut_iou(inpath, bbox, outpath):
    img = cv2.imread(inpath)
    iouimg = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    cv2.imwrite(outpath, iouimg)