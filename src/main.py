import torch
import torch.utils.data as Data
from Dataloader import ImgDataset
from Train_model import train_model
from Model.Draw_plot import draw_plot


def get_device():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print('GPU State:', device)

    return device


if __name__ == "__main__":
    device_0 = get_device()

    train_path = "Data/train"
    train_datasets = ImgDataset(train_path, isTrain=True)

    test_path = "Data/test"
    test_datasets = ImgDataset(test_path)

    train_loader = Data.DataLoader(
        dataset=train_datasets,
        batch_size=6,
        shuffle=True,
        num_workers=2
    )

    # cnn = CNN().to(device_0)
    # # print(cnn)
    # optimizer = torch.optim.Adam(cnn.parameters(), lr=1e-3)
    # loss_func = nn.CrossEntropyLoss()

    EPOCH = 30
    history = train_model(device=device_0, EPOCH=EPOCH,
                          train_loader=train_loader, test_datasets=test_datasets)
    # torch.save(cnn.state_dict(), 'out/cnn-model.pkl')
    (train_loss_ls, train_acc_ls, test_acc_ls) = history
    draw_plot(EPOCH, train_loss_ls, train_acc_ls,
              test_acc_ls, save_path='out/')
