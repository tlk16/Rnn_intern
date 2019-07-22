import torch
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from CNN_input import neuronSet
from torch.utils.data import DataLoader

data_dir = {'train': ('../png_data/png_v1/train/interneuron', '../png_data/png_v1/train/principal cell'),
            'test': ('../png_data/png_v1/test/interneuron', '../png_data/png_v1/test/principal cell')}

num_classes = len(data_dir['train'])
cudaFlag = torch.cuda.is_available()

lr = 1e-4
epochs = 20
batch_size = 100
trained = True

trainset = neuronSet('train', data_dir=data_dir)
testset = neuronSet('test', data_dir=data_dir)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=True)

if trained:
    model = torch.load('model')
else:
    model = models.resnet18(pretrained=True)
    # for parameter in model.parameters():
    #     parameter.requires_grad = False
    model.fc = nn.Linear(512, num_classes)
print(model)
fc_parameters = model.fc.parameters()
l = list(map(id, model.fc.parameters()))
conv_parameters = (parameter for parameter in model.parameters() if id(parameter) not in l)

if cudaFlag:
    model.cuda()
    model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])

def get_accuracy(logit, target, batch_size):
    # print(torch.max(logit, dim=1)[1])
    corrects = (torch.max(logit, dim=1)[1].view(target.size()).data == target.long().data).sum() #虽然没有经过softmax,但是softmax是单调的
    accuracy = 100.0 * corrects / batch_size
    return accuracy.item()

# 以较大学习率训练输出层，以较小学习率训练前面各层
optimizer = optim.Adam([{'params': fc_parameters, 'lr': lr}, {'params': conv_parameters, 'lr': lr/10}])
criterion = nn.CrossEntropyLoss()
for epoch in range(epochs):
    for phase in ['train', 'test']:
        epoch_accuracy = 0
        if phase == 'train':
            model.train()
            loader = trainloader
        else:
            model.eval()
            loader = testloader
        for i, batch in enumerate(loader):
            optimizer.zero_grad()
            data, labels = batch
            if data.shape[0] < batch_size:
                continue
            if cudaFlag:
                data, labels = data.cuda(), labels.cuda()

            outputs = model(data)
            accuracy = get_accuracy(outputs, labels, batch_size)
            loss = criterion(outputs, labels.long())
            print('batch_', i, accuracy, loss)
            epoch_accuracy = (epoch_accuracy * i + accuracy) / (i + 1)
            if phase == 'train':
                loss.backward()
                optimizer.step()
        print(epoch_accuracy)
    torch.save(model.module, 'model')
    # 86.49% in 'model', the best was 87.87%

    # 生成一些假的神经元，生成对抗
    # 先训练一个判别模型，
    # 然后做一个生成模型

    # 多分类
    # 传统特征分类

    # 应该查看tree rnn的梯度传播，考虑树太大时候的0梯度问题

