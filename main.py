import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from matplotlib.lines import Line2D
from matplotlib.text import Text
from torch import nn
from torch.utils.data import DataLoader
from dataset import MyDataset
from heatmap import Heatmap
from nn import Network, TorchNetwork
from nndraw import draw_network

device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"
print(f"Using {device} device")


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.train()
    total_loss = 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        total_loss += loss.item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return total_loss / num_batches


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    total_loss, total_correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            total_loss += loss_fn(pred, y).item()
            total_correct += (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()
    return total_loss / num_batches, total_correct / size


heatmap: Heatmap = None
img_heatmap: AxesImage = None
heatmap_first_time = True


def draw_heatmap(model, ax3: Axes, training_data, test_data):
    global img_heatmap
    with torch.no_grad():
        X = torch.from_numpy(heatmap.xdata).to(device)
        pred = model(X)
        heatdata = pred[:, 1].cpu().reshape(heatmap.height, heatmap.width)
        if img_heatmap is None:
            img_heatmap = ax3.imshow(heatdata)
        else:
            img_heatmap.set_data(heatdata)

    # draw data samples
    global heatmap_first_time
    if heatmap_first_time:
        heatmap_first_time = False
        # draw train samples
        max_heatmap_samples = 1000
        X, Y = training_data.get_data(max_heatmap_samples)
        cX, cY = heatmap.get_coord_of_data(X)
        pos_mask = Y[:, 1] == 1
        neg_mask = pos_mask == False
        ax3.plot(cX[pos_mask], cY[pos_mask], 'r.')
        ax3.plot(cX[neg_mask], cY[neg_mask], 'b.')
        # draw test samples
        X, Y = test_data.get_data(max_heatmap_samples)
        cX, cY = heatmap.get_coord_of_data(X)
        pos_mask = Y[:, 1] == 1
        neg_mask = pos_mask == False
        ax3.plot(cX[pos_mask], cY[pos_mask], 'm.')
        ax3.plot(cX[neg_mask], cY[neg_mask], 'c.')


stat_data = []
latest_test_accuracy = 0
stat_data_num_max = 300
is_first_draw_train_statistics = True
train_loss_line: Line2D = None
test_loss_line: Line2D = None
accuracy_text: Text = None


def draw_train_statistics(ax2:Axes, t, train_loss, test_loss, test_accuracy):
    global is_first_draw_train_statistics
    global train_loss_line
    global test_loss_line
    global accuracy_text
    global latest_test_accuracy
    global stat_data

    latest_test_accuracy = test_accuracy
    stat_data.append((train_loss, test_loss))
    if len(stat_data) > stat_data_num_max:
        del stat_data[0]

    if is_first_draw_train_statistics:
        is_first_draw_train_statistics = False
        train_loss_line, = ax2.plot([], [], lw=2, color='green', label='train loss')
        test_loss_line, = ax2.plot([], [], lw=2, color='red', label='test loss')
        ax2.legend()
        accuracy_text = ax2.text(0.2, 0.8, "000", transform=ax2.transAxes)

    stat_data_np = np.array(stat_data)
    ts = np.arange(t - len(stat_data) + 1, t+1)
    train_loss_line.set_data(ts, stat_data_np[:, 0])
    test_loss_line.set_data(ts, stat_data_np[:, 1])
    ax2.set_xlim(ts[0], ts[-1]+1)
    ymin, ymax = stat_data_np.min(), stat_data_np.max()
    ax2.set_ylim(min(ymin*0.9, ymin*1.1), max(ymax*0.9, ymax*1.1))
    accu_str = "accuracy=%.2f" % (latest_test_accuracy * 100)
    accuracy_text.set_text(accu_str)


is_window_closed = False


def on_close(event):
    global is_window_closed
    is_window_closed = True
    print("window closed!")


def create_model():
    model = Network()
    model.addlayer(2)
    model.addlayer(8)
    model.addlayer(8)
    model.addlayer(4)
    model.addlayer(2)
    return model
    # return TorchNetwork()


def main():
    global heatmap
    fig: Figure = plt.figure(figsize=(16, 9))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_aspect(1)
    ax2: Axes = fig.add_axes([0.02, 0.79, 0.2, 0.2])
    ax3: Axes = fig.add_axes([0.8, 0.7, 0.2, 0.3])
    # ax3: Axes = fig.add_axes([0.5, 0.5, 0.5, 0.5])
    plt.show(block=False)
    fig.canvas.mpl_connect('close_event', on_close)

    train_sample_num = 400
    test_sample_num = 100
    sample_noise = 0.1
    batch_size = 20
    training_data = MyDataset(train_sample_num, sample_noise)
    test_data = MyDataset(test_sample_num, sample_noise)
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    xmin, xmax = training_data.get_x_range()
    heatmap = Heatmap(100, 100, xmin[0], xmax[0], xmin[1], xmax[1])

    model = create_model().to(device)
    draw_network(ax, model, True)

    lr = 1e-2
    momentum = 0
    dampening = 0
    weight_decay = 0.001
    nesterov = False
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum,
                                dampening=dampening, weight_decay=weight_decay, nesterov=nesterov)
    loss_fn = nn.CrossEntropyLoss()

    epochs = 3000
    for t in range(epochs):
        train_loss = train(train_dataloader, model, loss_fn, optimizer)
        test_loss, test_accuracy = test(test_dataloader, model, loss_fn)
        draw_network(ax, model, False)
        draw_heatmap(model, ax3, training_data, test_data)
        draw_train_statistics(ax2, t, train_loss, test_loss, test_accuracy)
        plt.pause(0.01)
        if is_window_closed:
            break

    ax.autoscale_view(scalex=True, scaley=True)
    plt.show(block=True)
    print("Done!")


if __name__ == '__main__':
    main()
