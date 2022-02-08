import torch
import torchvision.datasets
from torch.utils.data import DataLoader

import transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#模型结构
class SimpleNet(torch.nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(3,32,5,padding=2)
        self.max_pool1 = torch.nn.MaxPool2d(2)
        self.conv2 = torch.nn.Conv2d(32,32,5,padding=2)
        self.max_pool2 = torch.nn.MaxPool2d(2)
        self.conv3 = torch.nn.Conv2d(32,64,5,padding=2)
        self.max_pool3 = torch.nn.MaxPool2d(2)
        self.linear1 = torch.nn.Linear(64*4*4,64)
        self.linear2 = torch.nn.Linear(64,10)
        self.flatten1 = torch.nn.Flatten(start_dim=1,end_dim=-1)

    def forward(self,x):
        x = self.conv1(x)
        x = self.max_pool1(x)
        x = self.conv2(x)
        x = self.max_pool2(x)
        x = self.conv3(x)
        x = self.max_pool3(x)
        x = self.flatten1(x)
        x = self.linear1(x)
        x = self.linear2(x)
        return x




#准备数据
train_data = torchvision.datasets.CIFAR10(root ='../../data/cifar10',train=True, transform= torchvision.transforms.ToTensor(),download=True)
test_data = torchvision.datasets.CIFAR10(root ='../../data/cifar10',train=False, transform= torchvision.transforms.ToTensor(),download=True)

train_data_size = len(train_data)
test_data_size = len(test_data)

print("训练数据的大小为:{}".format(train_data_size))
print("测试数据的大小为:{}".format(test_data_size))

train_loader = DataLoader(train_data,batch_size=64,shuffle=True,drop_last=True)
test_loader = DataLoader(test_data,batch_size= 64,shuffle=True)

simplenet = SimpleNet()
simplenet.to(device)

# 设置损失函数
loss_f = torch.nn.CrossEntropyLoss()
loss_f.to(device)

# 设置优化器
SGD_opt = torch.optim.SGD(simplenet.parameters(),lr=0.01)



#设置训练参数
epoch= 100
total_train_step = 0
total_test_step = 0



if __name__ == '__main__':
    for e in range(epoch):
        print("第{}轮训练开始".format(e+1))

        simplenet.train()
        for data in train_loader:
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            output = simplenet(imgs)
            loss = loss_f(output,targets)

            #更新参数
            SGD_opt.zero_grad()
            loss.backward()
            SGD_opt.step()
            total_train_step += 1
            if total_train_step % 100 == 0:
                print("迭代次数{},loss:{}".format(total_train_step,loss.item()))

        total_test_loss = 0
        total_acc = 0

        simplenet.eval()
        with torch.no_grad():
            for data in test_loader:
                imgs, targets = data
                imgs = imgs.to(device)
                targets = targets.to(device)
                outputs = simplenet(imgs)
                loss = loss_f(outputs,targets)
                acc = ((outputs.argmax(1) == targets).sum())/len(targets)

                total_acc += acc
                total_test_loss = total_test_loss + loss


            print("测试集的loss为{},准确率为{}".format(total_test_loss,total_acc/len(test_loader)))


    torch.save(simplenet.state_dict(),"../../cifar10model/cifar10.pth")














    # test_data = torch.ones((64,3,32,32))
    # simple_model = SimpleNet()
    # output = simple_model(test_data)
    # print(output.shape)
