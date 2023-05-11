from torch_geometric.nn import GCNConv
import torch
import torch.nn.functional as F
from tqdm import tqdm, trange
import pickle
import logging
import sys
import argparse
import os
import json

import torch_xla.core.xla_model as xm

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

num_features = 1433
num_classes = 7

epochs = 200
dim = 16

loss = 999.0
train_acc = 0.0
test_acc = 0.0


def load_data(path):
    with open(path, "rb") as f:
        data = pickle.load(f)

    # Since there is only 1 graph, the train/test split is done by masking regions of the graph. We split the last 500+500 nodes as val and test, and use the rest as the training data.
    data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.train_mask[:data.num_nodes - 1000] = 1
    data.val_mask = None
    data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.test_mask[data.num_nodes - 500:] = 1

    return data


class Net(torch.nn.Module):
    def __init__(self, num_features, dim=16, num_classes=1):
        super(Net, self).__init__()
        self.conv1 = GCNConv(num_features, dim)
        self.conv2 = GCNConv(dim, num_classes)

    def forward(self, x, edge_index, data=None):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


def test(model, data):
    model.eval()
    logits, accs = model(data.x, data.edge_index, data), []
    for _, mask in data('train_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs


def save_model(model, model_dir):
    logger.info("Saving the model.")
    path = os.path.join(model_dir, "checkpoints/checkpoint.pt")

    os.makedirs(model_dir, exist_ok=True)
    checkpoint = {'state_dict': model.state_dict()}

    xm.save(checkpoint, path)


def train(args):
    logger.info(f"cuda is available: {torch.cuda.is_available()}")
    logger.info(f"torch version: {torch.__version__}")

    data = load_data(os.path.join(args.data_dir, "data.pkl"))

    device = xm.xla_device()
    model = Net(num_features=num_features, dim=dim, num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-3)

    t = trange(epochs, desc="Stats: ", position=0)

    for epoch in t:
        model.train()

        loss = 0

        data = data.to(device)
        optimizer.zero_grad()
        log_logits = model(data.x, data.edge_index, data)

        # Since the data is a single huge graph, training on the training set is done by masking the nodes that are not in the training set.
        loss = F.nll_loss(log_logits[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        xm.mark_step()

        # validate
        train_acc, test_acc = test(model, data)
        train_loss = loss

        t.set_description('[Train_loss:{:.6f} Train_acc: {:.4f}, Test_acc: {:.4f}]'.format(loss, train_acc, test_acc))

    save_model(model, args.model_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--hosts", type=list, default=json.loads(os.environ["SM_HOSTS"]))
    parser.add_argument("--current-host", type=str, default=os.environ["SM_CURRENT_HOST"])
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--data-dir", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument("--num-gpus", type=int, default=os.environ["SM_NUM_GPUS"])

    train(parser.parse_args())



