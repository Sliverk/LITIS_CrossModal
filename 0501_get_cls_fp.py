import numpy as np
import sys
from utils.iou import kitti_eval
from utils.load import load_result_files, load_gt_annos
from utils.cut import cut_iou
import os
from multiprocessing import Pool


def main():

    RESULT_FILES_PATH=sys.argv[1]
    if RESULT_FILES_PATH[-1] != '/': RESULT_FILES_PATH = RESULT_FILES_PATH + '/'
    
    
    GT_ANNOS_PATH = './data/kitti_3d/training/label_2/'
    DATA_SPLIT_FILE = './data/kitti_3d/split/val.txt'
    IMG_PATH = './data/kitti_3d/training/image_2/'
    OUT_PATH = 'data/vis/' + RESULT_FILES_PATH.split('/')[-3] + '/' + RESULT_FILES_PATH.split('/')[-2] + '/'

    if not os.path.exists(OUT_PATH): os.makedirs(OUT_PATH)

    CLASSES = ('Car', 'Pedestrian', 'Cyclist')

    with open(DATA_SPLIT_FILE, 'r') as f:
        lines = f.readlines()
    filelist = [int(line) for line in lines]

    dt_annos = load_result_files(RESULT_FILES_PATH, filelist)
    gt_annos = load_gt_annos(GT_ANNOS_PATH, filelist)
    
    rets = kitti_eval(gt_annos, dt_annos, CLASSES)
    overlaps, parted_overlaps, total_dt_num, total_gt_num = rets
    # print(ap_result_str)
    # print(ap_dict)

    pool = Pool(4)
    res = []
    # dt
    count = 0
    badcount = 0
    for ix, ovp in enumerate(overlaps):
        # gt
        for jx, line in enumerate(ovp):
            # 
            flag = False
            maxsc = 0
            for px, sc in enumerate(line):
                if sc > maxsc: maxsc = sc
                if sc >= 0.7: 
                    count += 1
                    flag = True
                    break
            if not flag:
                maxsc = int(maxsc*100)
                filename ='%06d' %filelist[ix]
                imgpath = IMG_PATH + filename + '.png'
                outpath = OUT_PATH + filename + f'_{badcount}_{maxsc}' + '.png'
                bbox = np.array(dt_annos[ix]['bbox'][jx]).astype(np.int)
                pool.apply_async(cut_iou, args=(imgpath, bbox, outpath))
                # cut_iou(imgpath, bbox, outpath)
                badcount += 1
    pool.close()
    pool.join()
    print(count)
    print(badcount)

if __name__ == '__main__':
    main()
