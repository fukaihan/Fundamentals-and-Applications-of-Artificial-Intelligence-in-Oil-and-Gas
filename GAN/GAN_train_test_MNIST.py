import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from tqdm import tqdm
import numpy as np

# %% 1 判断设备训练方式(GPU or CPU)
device = ('cuda' if torch.cuda.is_available() else 'cpu')


# %% 2 生成器模型
class generator(nn.Module):
    # initializers
    def __init__(self, input_size=32, out_size=28 * 28):
        super(generator, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(self.fc1.out_features, 512)
        self.fc3 = nn.Linear(self.fc2.out_features, 1024)
        self.fc4 = nn.Linear(self.fc3.out_features, out_size)

    # forward method
    def forward(self, input):
        x = F.leaky_relu(self.fc1(input), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = torch.tanh(self.fc4(x))

        return x


# %% 3 判别器模型
class discriminator(nn.Module):
    # initializers
    def __init__(self, input_size=32, out_size=10):
        super(discriminator, self).__init__()
        self.fc1 = nn.Linear(input_size, 1024)
        self.fc2 = nn.Linear(self.fc1.out_features, 512)
        self.fc3 = nn.Linear(self.fc2.out_features, 256)
        self.fc4 = nn.Linear(self.fc3.out_features, out_size)

    # forward method
    def forward(self, input):
        x = F.leaky_relu(self.fc1(input), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = F.dropout(x, 0.3)
        x = torch.sigmoid(self.fc4(x))

        return x


# %% 4 训练损失显示
def show_train_hist(hist, show=False):
    x = np.arange(1, len(hist['D_losses']) + 1)

    y1 = hist['D_losses']
    y2 = hist['G_losses']

    plt.plot(x, y1, label='D_loss')
    plt.plot(x, y2, label='G_loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.legend(loc=4)
    plt.tight_layout()

    if show:
        plt.show()
    else:
        plt.close()

# %% 5 模型训练
def Train(batch_size, lr, train_epoch, G_input_size, G_output_size, D_input_size, D_output_size):
    # data_loader
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.5), std=(0.5))])
    data = datasets.MNIST('data', train=True, download=True, transform=transform)
    data_index = torch.nonzero(data.targets == 0).squeeze()
    data = data.data[data_index].float()
    train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)

    # network
    G = generator(input_size=G_input_size, out_size=G_output_size).to(device)
    D = discriminator(input_size=D_input_size, out_size=D_output_size).to(device)

    # Binary Cross Entropy loss
    BCE_loss = nn.BCELoss()

    # Adam optimizer
    G_optimizer = optim.Adam(G.parameters(), lr=lr)
    D_optimizer = optim.Adam(D.parameters(), lr=lr)

    train_hist = {}
    train_hist['D_losses'] = []
    train_hist['G_losses'] = []
    for epoch in range(train_epoch):
        D_losses = []
        G_losses = []
        for x_ in tqdm(train_loader):
            # train discriminator D
            D.zero_grad()

            x_ = x_.view(-1, G_output_size)
            mini_batch = x_.size()[0]
            y_real_ = torch.ones(mini_batch)
            y_fake_ = torch.zeros(mini_batch)
            x_, y_real_, y_fake_ = Variable(x_.to(device)), Variable(y_real_.to(device)), Variable(y_fake_.to(device))
            D_result = D(x_).squeeze()
            D_real_loss = BCE_loss(D_result, y_real_)

            z_ = torch.randn((mini_batch, G_input_size))
            z_ = Variable(z_.to(device))
            G_result = G(z_)
            D_result = D(G_result).squeeze()
            D_fake_loss = BCE_loss(D_result, y_fake_)

            D_train_loss = D_real_loss + D_fake_loss
            D_train_loss.backward()
            D_optimizer.step()

            D_losses.append(D_train_loss.item())

            # train generator G
            G.zero_grad()

            z_ = torch.randn((mini_batch, G_input_size))
            y_ = torch.ones(mini_batch)
            z_, y_ = Variable(z_.to(device)), Variable(y_.to(device))
            G_result = G(z_)
            D_result = D(G_result).squeeze()
            G_train_loss = BCE_loss(D_result, y_)
            G_train_loss.backward()
            G_optimizer.step()

            G_losses.append(G_train_loss.item())

        print('[%d/%d]: loss_d: %.3f, loss_g: %.3f' % ((epoch + 1), train_epoch,
                                                       torch.mean(torch.FloatTensor(D_losses)),
                                                       torch.mean(torch.FloatTensor(G_losses))))
        train_hist['D_losses'].append(torch.mean(torch.FloatTensor(D_losses)))
        train_hist['G_losses'].append(torch.mean(torch.FloatTensor(G_losses)))

    print("Training finish!... save training model")
    torch.save(G, "GAN_generator_param.pkl")

    show_train_hist(train_hist, show=True)
    
    return G


if __name__ == '__main__':
    # training parameters
    batch_size = 228
    lr = 0.00001
    train_epoch = 30
    G_input_size = 200
    G_output_size = 28 * 28
    D_input_size = G_output_size
    D_output_size = 1

    G=Train(batch_size, lr, train_epoch, G_input_size, G_output_size, D_input_size, D_output_size)
    
    z_ = torch.randn((1, G_input_size))
    z_ = z_.cuda()
    G_result = G(z_)
    G_result = G_result.reshape(28, 28).cpu().detach().numpy()
    G_result = np.where(G_result > 0.5, 1, 0)

    plt.imshow(G_result, cmap='gray')
    plt.axis('off')
    plt.show()
