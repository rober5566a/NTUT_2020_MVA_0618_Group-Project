import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
from Dataloader import ImgDataset


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # self.nn = nn

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=50,
                      kernel_size=(3, 3), stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)))

        self.conv2 = nn.Sequential(
            nn.Conv2d(50, 100, (3, 3), 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)))

        self.ln1 = nn.Linear(72600, 500)
        self.out = nn.Linear(500, 3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.ln1(x)
        output = self.out(x)

        return output

    # def forward(self, x):
    #     x = self.conv1(x)
    #     x = self.conv2(x)
    #     x = x.view(x.size(0), -1)
    #     self.out = nn.Linear(x.size(1), 10).to(device_0)
    #     output = self.out(x)

    #     return output


def get_device():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print('GPU State:', device)

    return device


def train_model(EPOCH, train_loader, test_datasets):
    test_datas = torch.tensor(test_datasets.datas, dtype=torch.float32)
    test_labels = torch.tensor(test_datasets.labels, dtype=torch.int64)
    test_datas = Variable(test_datas).to(device=device_0)
    test_labels = Variable(test_labels).to(device=device_0)
    for epoch in range(EPOCH):
        print("EPOCH: ", epoch)

        total = 0
        train_acc = 0
        for step, (datas, labels) in enumerate(train_loader):
            datas = torch.tensor(datas, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
            b_datas = Variable(datas).to(device=device_0)
            b_labels = Variable(labels).to(device=device_0)

            output = cnn.forward(b_datas)
            loss = loss_func(output, b_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pred_train_labels = torch.max(output, 1)[1].data.squeeze()

            total += b_labels.size(0)

            if step % 50 == 0:
                test_output = cnn(test_datas)
                pred_test_labels = torch.max(test_output, 1)[1].data.squeeze()

                # train_acc
                train_num_right = int(sum(pred_train_labels == b_labels))
                train_acc = train_num_right / test_labels.size(0)

                test_num_right = int(sum(pred_test_labels == test_labels))
                test_acc = test_num_right / test_labels.size(0)
                # pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
                print('Batch Size: {} | train_acc: {:5f} | train_loss: {:5f} | test_acc: {:5f}'.format(
                    step, test_acc, loss, test_acc))

        # test_output = cnn(test_datas[:10])
        # pred_y = torch.max(test_output, 1)[1].data.squeeze()
        # print(pred_y, 'prediction number')
        # print(test_labels[:10], 'real number')


if __name__ == "__main__":
    device_0 = get_device()

    train_path = "Data/train"
    train_datasets = ImgDataset(train_path)

    test_path = "Data/test"
    test_datasets = ImgDataset(test_path)

    train_loader = Data.DataLoader(
        dataset=train_datasets,
        batch_size=5,
        shuffle=True,
        num_workers=2
    )

    cnn = CNN().to(device_0)
    # print(cnn)
    optimizer = torch.optim.Adam(cnn.parameters(), lr=1e-3)
    loss_func = nn.CrossEntropyLoss()

    EPOCH = 100
    train_model(EPOCH, train_loader, test_datasets)
