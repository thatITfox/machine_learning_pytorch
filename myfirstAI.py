import torch
import torchvision
import matplotlib.pyplot as plt
from torchvision import transforms, datasets
import torch.nn as nn
import torch. nn.functional as f
import torch.optim as optim

# download the data set and define it
train = datasets.MNIST("", train=True, download=True,
                       transform = transforms.Compose([transforms.ToTensor()]))

test = datasets.MNIST("", train=False, download=True,
                       transform = transforms.Compose([transforms.ToTensor()]))

# setup the data set
trainset = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)
testset = torch.utils.data.DataLoader(test, batch_size=10, shuffle=True)

load_model = True

# the neural network
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 10)

    def forward(self, x):
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        x = f.relu(self.fc3(x))
        x = self.fc4(x)
        return f.log_softmax(x, dim=1)

# def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
#     print("saving checkpoint")
#     torch.save(state, filename)

# def load_checkpoint(checkpoint):
#     print("loading checkpoint")
#     net.load_state_dict(checkpoint['state_dict'])
#     optimizer.load_state_dict(checkpoint['state_dict'])

# the network object
net= Net()
# print(net)

optimizer = optim.Adam(net.parameters(), lr=0.001)

EPOCHS = 3

# if load_model:
#     load_checkpoint(torch.load("my_checkpoint.pth.tar"))

for epochs in range(EPOCHS):
    # if epochs % 3 == 0:
    #     checkpoint = {'state_dict': net.state_dict(), 'optimizer': optimizer.state_dict()}
    #     save_checkpoint(checkpoint)

    for data in trainset:
        # data is a batch of features and labels
        X, y = data
        net.zero_grad()
        output = net(X.view(-1, 28*28))
        loss = f.nll_loss(output, y)
        loss.backward()
        optimizer.step()
    print(loss)

correct = 0
total = 0

with torch.no_grad():
    for data in trainset:
        X, y = data
        output = net(X.view(-1, 784))
        for idx, i in enumerate(output):
            if torch.argmax(i) == y[idx]:
                correct += 1
            total += 1

print('accuracy:', round(correct/total, 3))
print(torch.argmax(net(X[0].view(-1, 784))))
plt.imshow(X[0].view(28, 28))
plt.show()