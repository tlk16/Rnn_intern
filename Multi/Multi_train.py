import torch
from Multi_input import multiSet
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from TRNN.TRNN_train import lines2tree

# 主要问题：换数据时要换input很麻烦
# 1. 五种数据类型： swc, swc_fake, png, png_fake, 3D
# 2. 三种
# 3. 细粒度分类

trnn_model_filename = '../TRNN/model_v0'
cnn_model_filename = '../CNN/model'
num_classes = 5
batch_size = 50
epoches = 10
lr = 1e-4

trnn_model = torch.load(trnn_model_filename)
cnn_model = torch.load(cnn_model_filename)
print(trnn_model)
print(cnn_model)
trnn_feature_size = list(trnn_model.children())[-1].weight.shape[1]
cnn_feature_size = list(cnn_model.children())[-1].weight.shape[1]
cnn_model = nn.Sequential(*list(cnn_model.children())[:-1])

fc = nn.Linear(trnn_feature_size + cnn_feature_size, num_classes)
fc.cuda()

trainset = multiSet('train')
train_loader = DataLoader(trainset, shuffle=True, batch_size=batch_size)
testset = multiSet('test')
test_loader = DataLoader(testset, shuffle=True, batch_size=batch_size)

cudaFlag = torch.cuda.is_available()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(fc.parameters(), lr=lr)

def get_accuracy(logit, target, batch_size):
    # print(torch.max(logit, dim=1)[1])
    corrects = (torch.max(logit, dim=1)[1].view(target.size()).data == target.long().data).sum() #虽然没有经过softmax,但是softmax是单调的
    accuracy = 100.0 * corrects / batch_size
    return accuracy.item()

for epoch in range(epoches):
    for phase in ['train', 'test']:
        epoch_accuracy = 0
        if phase == 'train':
            loader = train_loader
        else:
            loader = test_loader
        for i, batch in enumerate(loader):
            swc_data, tree_len, png_data, labels = batch
            optimizer.zero_grad()
            swc_outputs = torch.zeros(batch_size, 128)
            if cudaFlag: swc_data, tree_len, png_data, labels, swc_outputs = swc_data.cuda(), tree_len.cuda(), png_data.cuda(), labels.cuda(), swc_outputs.cuda()
            if swc_data.shape[0] < batch_size:
                continue
            labels = labels.resize(batch_size)
            png_outputs = cnn_model(png_data)
            for j in range(batch_size):
                output = trnn_model.tree_rnn(lines2tree((swc_data[j][0:tree_len[j]]).float()))[3].unsqueeze(0)
                swc_outputs[j] = output
            png_outputs = png_outputs.squeeze()
            outputs = torch.cat((png_outputs, swc_outputs), 1)
            outputs = fc(outputs)   # 应该是batch_size * 5
            accuracy = get_accuracy(outputs, labels, batch_size)
            loss = criterion(outputs, labels.long())
            print('batch ', i, accuracy, loss)
            epoch_accuracy = (epoch_accuracy * i + accuracy) / (i + 1)
            if phase == 'train':
                loss.backward()
                optimizer.step()
        print(phase, epoch_accuracy)
    torch.save(fc, 'fc')
