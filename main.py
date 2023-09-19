import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from model import resnet18


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# model성능을 평가하기 위해test이라는 함수를 생성
def eval():

    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

# model을 학습시키기 위해  train이라는 함수를 생성
def train():
    for epoch in range(100):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    print('Finished Training')



if __name__ =='__main__':

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    # --batch_size--
    batch_size = 16 # batch size 설정
    # --------------
    # -----------------------------setup device----------------------------
    gpu_id = '1' # GPU로 학습 시키기 위해 device 설정
    device = torch.device("cuda:{}".format(gpu_id) if torch.cuda.is_available() else "cpu")
    # ---------------------------------------------------------------------

    #-------------------------------------dataset_download-------------------------------------------------
    # train, test데이터셋을 다운받고 데이터를 load해옴
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)   # 32*32*3
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    # ------------------------------------------------------------------------------------------------------

    # ----------------------------------------classes-----------------------------------------
    # label에 대한 class
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    # -----------------------------------------------------------------------------------------

    # --network_setup--
    # network 설정
    net = resnet18()
    net.to(device)
    # -----------------


    # ------------loss, optimizer, scheduler_setup--------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    # optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay =5e-5) # optimizer를 Adam으로 변경해보기 위함
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.1) # scheduler를 추가
    # -----------------------------------------------------------------


    train()

    # ./checkpoints/cifar_net.py에 weights를 저장
    path = './checkpoints/cifar_net.pth'
    torch.save(net.state_dict(), path)
    # ------------------------------------------

    net = resnet18
    # 저장된 weight를 불러와서 test를 함
    net.load_state_dict(torch.load(path))
    eval()
    # ------------------------------------



