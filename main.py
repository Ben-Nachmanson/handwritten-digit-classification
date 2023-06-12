import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms


# transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,),(0.5,))])

# download and load the training data
trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# download and load the test data
testset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

import matplotlib.pyplot as plt
import numpy as np
# show an image from the MNIST dataset
#plt.imshow(trainset[0][0].numpy().squeeze(), cmap='gray')


model = Network()

criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr = 0.03)

epochs = 20
losses = []
for e in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        optimizer.zero_grad()
        
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    else:
        average_loss = running_loss / len(trainloader)
        print(f"Training loss: {average_loss}")
        losses.append(average_loss)

plt.figure(figsize=(10,5))
plt.plot(range(epochs), losses)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss over Time')
#plt.show()

correct_count, all_count = 0, 0
for images, labels in testloader:
    for i in range(len(labels)):
        img = images[i].view(1,784)
        with torch.no_grad():
            logps = model(img)
        ps = torch.exp(logps)
        probab = list(ps.numpy()[0])
        pred_label = probab.index(max(probab))
        true_label = labels.numpy()[i]
        if(true_label == pred_label):
            correct_count+=1
        all_count +=1
print("Number of Images Tests =", all_count)
print("\nModel Accuracy =", (correct_count/all_count))

torch.save(model.state_dict(), 'trained_model.pth')

