import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import sys


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

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Dataloader
    DATA_DIR = sys.argv[1] if len(sys.argv)>1 else './data/kitti_3d/classification/210104_2back1fore/'
    train_dir = f'{DATA_DIR}/train/'
    val_dir = f'{DATA_DIR}/val/'

    transf = transforms.Compose([
        transforms.Resize((64,64)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],std = [0.229, 0.224, 0.225])
        transforms.Normalize(mean=[0.5,0.5,0.5],std = [0.5,0.5,0.5])
    ])

    trainset = datasets.ImageFolder(
        train_dir, 
        transforms.Compose([
        transforms.Resize((64,64)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],std = [0.229, 0.224, 0.225])
        transforms.Normalize(mean=[0.5,0.5,0.5],std = [0.5,0.5,0.5])
    ]))
    testset = datasets.ImageFolder(
        val_dir,
        transforms.Compose([
        transforms.Resize((64,64)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],std = [0.229, 0.224, 0.225])
        transforms.Normalize(mean=[0.5,0.5,0.5],std = [0.5,0.5,0.5])
    ]))
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=True, num_workers=2)
    
    # Network
    model = Net()
    model = model.to(device)
    # Train
    criterion = nn.CrossEntropyLoss()
    criterion.to(device=device)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    model_save_path = './workdir/210104/'

    print('Begin to train')
    for epoch in range(40):
        sum_loss = 0.0
        total = 0.0
        correct = 0.0
        model.train()
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # maxk = max((1,5))
            sum_loss += loss.item()
            label_resize = labels.view(-1,1)
            _, predicted = outputs.topk(1, 1, True, True)
            total += labels.size(0)
            correct += torch.eq(predicted, label_resize).cpu().sum().float().item()
            if i == 2200:
                print(f'[epoch:{epoch}, iter:{i}] Loss:{sum_loss/(i+1)} | Acc:{100.*correct/total}')
        torch.save(model.state_dict(), model_save_path+'kitti_%02d.pth' %(epoch+1))

        print('Testing...')
        with torch.no_grad():
            correct = 0
            model.eval()
            total = 0
            for data in testloader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)

                # maxk = max((1,5))
                label_resize = labels.view(-1,1)
                _, predicted = outputs.topk(1, 1, True, True)
                total += labels.size(0)
                correct += torch.eq(predicted, label_resize).cpu().sum().float().item()

                # y_predict.append(predicted)
                # y_true.append(labels)
            print(f'Test split accuracy: {100*correct/total}')
    # Test

if __name__ == '__main__':
    main()