import numpy as np
import sys
import pathlib


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
    # ('name', None),
    # ('truncated', -1),
    # ('occluded', -1),
    # ('alpha', -10),
    # ('bbox', None),
    # ('dimensions', [-1, -1, -1]),
    # ('location', [-1000, -1000, -1000]),
    # ('rotation_y', -10),
    # ('score', 0.0),
    # for res in filelist:
    #     with open(root_path+res+'.txt', 'r') as f:
    #         records = f.readlines()
    #     if records == None:
    #         recditc['name'] = None
    #         recditc['truncated'] = -1
    #         recditc['occluded'] = -1
    #         recditc['alpha'] = -10
    #         recditc['bbox'] = None
    #         recditc['dimensions'] = [-1, -1, -1]
    #         recditc['location'] = [-1000, -1000, -1000]
    #         recditc['rotation_y'] = -10
    #         recditc['score'] = 0.0
    #     else:
    #         for rec in records:
    return dt_annos

def load_gt_annos(root_path, filelist):
    gt_annos = get_label_annos(root_path, filelist)
    return gt_annos
