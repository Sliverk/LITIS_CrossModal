from utils.eval import kitti_eval
import pathlib
import numpy as np
import sys


#RESULT_FILES_PATH = './data/pred/'
#RESULT_FILES_PATH = './data/noblack_2cls_3fc_ep23/'

RESULT_FILES_PATH=sys.argv[1]
#RESULT_FILES_PATH='./data/results/pointpillars_lidar/'
#RESULT_FILES_PATH='./data/correct/210108T02_lidar/'
GT_ANNOS_PATH = './data/kitti_3d/training/label_2/'
DATA_SPLIT_FILE = './data/kitti_3d/split/val.txt'

def get_image_index_str(img_idx):
    return "{:06d}".format(img_idx)


def get_label_anno(label_path):
    annotations = {}
    annotations.update({
        'name': [],
        'truncated': [],
        'occluded': [],
        'alpha': [],
        'bbox': [],
        'dimensions': [],
        'location': [],
        'rotation_y': []
    })
    with open(label_path, 'r') as f:
        lines = f.readlines()
    # if len(lines) == 0 or len(lines[0]) < 15:
    #     content = []
    # else:
    content = [line.strip().split(' ') for line in lines]
    annotations['name'] = np.array([x[0] for x in content])
    annotations['truncated'] = np.array([float(x[1]) for x in content])
    annotations['occluded'] = np.array([int(x[2]) for x in content])
    annotations['alpha'] = np.array([float(x[3]) for x in content])
    annotations['bbox'] = np.array(
        [[float(info) for info in x[4:8]] for x in content]).reshape(-1, 4)
    # dimensions will convert hwl format to standard lhw(camera) format.
    annotations['dimensions'] = np.array(
        [[float(info) for info in x[8:11]] for x in content]).reshape(
            -1, 3)[:, [2, 0, 1]]
    annotations['location'] = np.array(
        [[float(info) for info in x[11:14]] for x in content]).reshape(-1, 3)
    annotations['rotation_y'] = np.array(
        [float(x[14]) for x in content]).reshape(-1)
    if len(content) != 0 and len(content[0]) == 16:  # have score
        annotations['score'] = np.array([float(x[15]) for x in content])
    else:
        annotations['score'] = np.zeros([len(annotations['bbox'])])
    return annotations

def get_label_annos(label_folder, image_ids=None):
    if image_ids is None:
        filepaths = pathlib.Path(label_folder).glob('*.txt')
        prog = re.compile(r'^\d{6}.txt$')
        filepaths = filter(lambda f: prog.match(f.name), filepaths)
        image_ids = [int(p.stem) for p in filepaths]
        image_ids = sorted(image_ids)
    if not isinstance(image_ids, list):
        image_ids = list(range(image_ids))
    annos = []
    label_folder = pathlib.Path(label_folder)
    for idx in image_ids:
        image_idx = get_image_index_str(idx)
        label_filename = label_folder / (image_idx + '.txt')
        annos.append(get_label_anno(label_filename))
    return annos

def load_result_files(root_path, filelist):
    dt_annos = get_label_annos(root_path, filelist)
    return dt_annos

def load_gt_annos(root_path, filelist):
    gt_annos = get_label_annos(root_path, filelist)
    return gt_annos

def main():
    CLASSES = ('Car', 'Pedestrian', 'Cyclist')

    with open(DATA_SPLIT_FILE, 'r') as f:
        lines = f.readlines()
    filelist = [int(line) for line in lines]

    result_files = load_result_files(RESULT_FILES_PATH, filelist)
    gt_annos = load_gt_annos(GT_ANNOS_PATH, filelist)
    
    ap_result_str, ap_dict = kitti_eval(gt_annos, result_files, CLASSES)
    # print(ap_result_str)
    # print(ap_dict)
if __name__ == '__main__':
    main()
