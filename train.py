import pandas as pd
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import time
import os
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
print(torch.cuda.get_device_name(device))

dataset_dir = "/home/sidney/Desktop/hw1/train"

train_tfms = transforms.Compose([transforms.Resize(256),
                                 transforms.CenterCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.RandomRotation(30),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406],
                                                      [0.229, 0.224, 0.225])])

dataset = torchvision.datasets.ImageFolder(
                                        root=dataset_dir, transform=train_tfms)
idx2class = {v: k for k, v in dataset.class_to_idx.items()}
valid_size = int(0.05*len(dataset))
train_size = len(dataset) - valid_size
train_dataset, valid_dataset = torch.utils.data.random_split(
                               dataset, [train_size, valid_size])

trainLoader = torch.utils.data.DataLoader(train_dataset, batch_size=8,
                                          shuffle=True, num_workers=2)
validLoader = torch.utils.data.DataLoader(valid_dataset, batch_size=8,
                                          shuffle=False, num_workers=2)

# training--------------------------------------------------------
model = models.resnet152(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 196)
model = model.cuda()

training_loss = []
testing_loss = []
correct_v = []
# Parameters
criterion = nn.CrossEntropyLoss()
lr = 0.001
epochs = 60
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

for epoch in range(epochs):
    running_loss = 0.0
    correct = 0
    correct_v = 0
    total = 0
    total_v = 0
    test_loss = 0.0
    # training data
    model.train()
    for times, data in enumerate(trainLoader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        my_lr_scheduler = torch.optim.lr_scheduler.StepLR(
                                         optimizer, step_size=20, gamma=0.9)
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    # validation data
    model.eval()
    for times, data in enumerate(validLoader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_v += labels.size(0)
        correct_v += (predicted == labels).sum().item()

    print('[%d/%d] train_loss:%.3f train_acc:%d%% valid_acc:%d%%'
          % (epoch+1, epochs, running_loss/2000,
             100*correct/total, 100*correct_v/total_v))
    correct_v = np.append(correct_v, 100*correct_v/total_v)
    training_loss = np.append(training_loss, running_loss/2000)
    testing_loss = np.append(testing_loss, test_loss/2000)

print('Finished Training')
torch.save(model, '/home/sidney/Desktop/hw1/model.pth')

# plot curvve------------------------------------
epoch = []
for i in range(1, epochs+2):
    epoch = np.append(epoch, i-1)
trainingloss = np.append(training_loss[0]+0.1, training_loss)
testingloss = np.append(testing_loss[0]+0.1, testing_loss)
plt.plot(epoch, trainingloss, color='blue')
plt.title('Training Loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()
plt.close()

plt.plot(epoch, testingloss, color='orange')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title("testing error")
plt.show()


# testing------------------------------------------------------------------

test_dir = '/home/sidney/Desktop/hw1/test'

test_tfms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224), transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
dataset_test = torchvision.datasets.ImageFolder(
                                            root=test_dir, transform=test_tfms)
testloader = torch.utils.data.DataLoader(dataset_test, batch_size=32,
                                         shuffle=False, num_workers=2)
with torch.no_grad():
    model.eval()
    image_names = []
    pred = []
    for index in testloader.dataset.imgs:
        image_names.append(Path(index[0]).stem)
    results = []
    file_names = []
    predicted_car = []
    predicted_class = []

    for inputs, labels in testloader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, pred = torch.max(outputs.data, 1)

        for i in range(len(inputs)):
            file_names.append(image_names[i])
            predicted_car.append(int(pred[i]))
results.append((file_names, predicted_car))
df_predict = pd.DataFrame({'id': image_names, 'label': predicted_car})
id_number = df_predict['label'].to_list()
id_class = list(idx2class.values())
out = []
for i in range(0, 5000):
    x = id_number[i]
    out.append(id_class[x])
df_predict['label'] = out
df_predict.to_csv('/home/sidney/Desktop/hw1/predictions.csv', index=False)



