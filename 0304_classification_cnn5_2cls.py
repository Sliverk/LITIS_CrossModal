import os
import sys
import glob

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

import pickle
from PIL import Image


def load_pred_info(FILE_ROOT, filelist):
    bbox2d = []
    for filename in filelist:
        filepath = FILE_ROOT + filename + '.txt'
        with open(filepath, 'r') as f:
            lines = f.readlines()
        bboxlist = []
        for line in lines:
            bboxdict = {}
            bbox = line.split(' ')[4:8]
            bbox = list(map(float, bbox))
            bbox = list(map(int, bbox))
            bboxdict['line'] = line
            bboxdict['bbox'] = bbox
            bboxdict['score'] = []
            bboxdict['class'] = 0
            bboxlist.append(bboxdict)
        bbox2d.append(bboxlist)
    return bbox2d


def get_part_image(imagepath, rois):
    imagerois = []
    img = Image.open(imagepath)
    for rec in rois:
        rec = rec['bbox']
        part = img.crop((rec[0],rec[1],rec[2],rec[3]))
        imagerois.append(part)
        # maxlen = max(rec[3]-rec[1], rec[2]-rec[0])
        # blankimg = Image.new('RGB', (maxlen,maxlen))
        # blankimg.paste(part, (0,0))
    return imagerois

def get_prediction(imglist, network):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_transform = transforms.Compose(
    [transforms.Resize((64,64)),
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.5,0.5,0.5],std = [0.5,0.5,0.5])])
    
    imagebatch = []
    for image in imglist:
        imagebatch.append(data_transform(image))
    imagebatch = torch.stack(imagebatch, 0)
    imagebatch = imagebatch.to(device)

    network.eval()
    with torch.no_grad():
        output = torch.squeeze(network(imagebatch))
        predict = torch.softmax(output, 0)
        # if 
        # print(predict)
        predict_cls = torch.argmax(predict,dim=1).cpu().numpy()
        # print(predict_cls)
    return predict, predict_cls

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.conv4 = nn.Conv2d(128, 256, 3)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(1024, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv4(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        # x = self.dropout1(x)
        x = torch.flatten(x, 1)
        # print(x.shape)
        x = self.fc1(x)
        x = F.relu(x)
        # x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

def load_network(modelpath):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    network = Net()    
    network = network.to(device)
    state_dict = torch.load(modelpath,map_location=torch.device(device))
    network.load_state_dict(state_dict)
    return network

def remove_fp(imageroot, filelist, predinfo, modelpath, output):
    fileindex = 0
    network = load_network(modelpath)
    for filename in filelist:
        imagepath = imageroot + '/' + filename + '.png'
        partimg = get_part_image(imagepath, predinfo[fileindex])
        scores, rets = get_prediction(partimg, network)
        outputfile = output + f'/{filename}.txt'
        outf = open(outputfile,'w')
        for x in range(len(predinfo[fileindex])):            
            if rets[x] == 0:
                line = predinfo[fileindex][x]['line']
                outf.write(line)
            # prepare for pickle
            predinfo[fileindex][x]['score'] = scores[x]
            predinfo[fileindex][x]['class'] = rets[x]
        fileindex += 1
        # if fileindex == 10: break
        if fileindex % 100 == 0: print(f'{fileindex} have been processed.')
    # # write to pickle
    # with open('./results0303.pkl','wb') as f:
    #     pickle.dump(predinfo, f)

def main():
    LOAD_MODE_FLAG = input('Load from pickle file y/N?:')
    LOAD_MODE = 'IMG'
    if LOAD_MODE_FLAG == 'Y' or LOAD_MODE_FLAG == 'y':
        LOAD_MODE = 'PKL'
    
    if LOAD_MODE == 'IMG':
        # Define Global variable
        KITTI_PATH = './data/kitti_3d/'
        MODEL_FILE = './workdir/210104T02/kitti_40.pth'

        THRESHOLD = 0.5
        SPLIT_FILE = f'{KITTI_PATH}/split/val.txt'
        IMAGE_PATH = f'{KITTI_PATH}/training/image_2/'

        INPUT_RESULT_PATH = './data/pred/'
        OUTPUT_RESULT_PATH = './data/correct/210104/'

        if not os.path.exists(OUTPUT_RESULT_PATH): os.makedirs(OUTPUT_RESULT_PATH)
        # SCORES_SAVE_FILE = './data/scores/201203T01EP120_result.pkl'
        # Load file list
        with open(SPLIT_FILE, 'r') as f:
            filelist = f.readlines()
        filelist = [x.strip() for x in filelist]
        # Load pred info
        pred_info = load_pred_info(INPUT_RESULT_PATH, filelist)
        # classification
        remove_fp(IMAGE_PATH,filelist,pred_info,MODEL_FILE,OUTPUT_RESULT_PATH)
        # # Get part image
        # partimgs = get_part_image(IMAGE_PATH, filelist, pred_info)
        # print(len(partimgs))
        # Classification

        # Save to pkl
        # Remove false positive
    elif LOAD_MODE == 'PKL':
        print('Not Implement')
    else: print('ERROR WITH LOAD SOURCE')

if __name__ == '__main__':
    main()