
import numpy as np
import torch
import torch.nn as nn

from .data import prepare_batch, gen_data_batch, results_converter


class myPTRNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_layer = nn.Embedding(10, 32)
        self.rnn = nn.RNN(64, 64, 2)
        self.dense = nn.Linear(64, 10)

    def forward(self, num1: torch.Tensor, num2: torch.Tensor):
        x1 = self.embed_layer(num1)
        x2 = self.embed_layer(num2)
        x = torch.cat((x1, x2), dim=2).transpose(0, 1)
        h = self.rnn(x)[0]
        y_pred = self.dense(h)
        return y_pred.transpose(0, 1).clone()


class myAdvPTRNNModel(nn.Module):
    def __init__(self):
        '''
        Please finish your code here.
        '''
        super().__init__()

    def forward(self, num1, num2):
        '''
        Please finish your code here.
        '''
        return logits


def compute_loss(logits, labels):
    losses = nn.CrossEntropyLoss()
    return losses(logits.view(-1, 10), labels.view(-1))


def train_one_step(model, optimizer, x, y, label, dev):
    model.train()
    optimizer.zero_grad()
    logits = model(torch.tensor(x, device=dev), torch.tensor(y, device=dev))
    loss = compute_loss(logits, torch.tensor(label, device=dev))

    # compute gradient
    loss.backward()
    optimizer.step()
    return loss.item()


def train(steps, model, optimizer, num_digits, dev, mp):
    loss = 0.0
    accuracy = 0.0
    for step in range(steps):
        datas = gen_data_batch(batch_size=64, start=0, end=5 * 10**num_digits)
        Nums1, Nums2, results = prepare_batch(*datas, mp, maxlen=num_digits + 2)
        loss = train_one_step(model, optimizer, Nums1,
                              Nums2, results, dev)
        if (step + 1) % 50 == 0:
            for o in list(zip(results, Nums1, Nums2))[:2]:
                print(o[1], o[2], o[0])
            print('step', step+1, ': loss', loss)

    return loss


def evaluate(model, num_digits, dev, mp):
    datas = gen_data_batch(batch_size=10000, start=5 * 10**num_digits + 1, end=10**(num_digits + 1))
    Nums1, Nums2, results = prepare_batch(*datas, mp, maxlen=num_digits + 2)
    with torch.no_grad():
        logits = model(torch.tensor(Nums1, device=dev), torch.tensor(Nums2, device=dev))
    logits = logits.cpu().numpy()
    pred = np.argmax(logits, axis=-1)
    res = results_converter(pred)
    real_ans = results_converter(results)

    for o in list(zip(real_ans, res, Nums1, Nums2))[:20]:
        print(o[2], o[3], o[0], o[1], o[0]==o[1])

    print('accuracy is: %g' % np.mean([o[0]==o[1] for o in zip(real_ans, res)]))


def pt_main(num_digits, mp, use_cuda=False):
    if use_cuda and not torch.cuda.is_available():
        print("(warn) Cuda not available!")

    if use_cuda and torch.cuda.is_available():
        dev = torch.device("cuda")
    else:
        dev = torch.device("cpu")

    model = myPTRNNModel().to(dev)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
    train(1000, model, optimizer, num_digits, dev, mp)
    evaluate(model, num_digits, dev, mp)


def pt_adv_main():
    '''
    Please finish your code here.
    '''
    pass
