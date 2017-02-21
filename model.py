import argparse
import random
import json
import spacy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable


class TreeLSTMCell(nn.Module):
    """A Child-Sum Tree-Long Short Term Memory cell.

    """
    def __init__(self, input_size, hidden_size):
        super(TreeLSTMCell, self).__init__()
        self.W_i = nn.Parameter(torch.Tensor(input_size,
                                             hidden_size).uniform_(-0.1, 0.1))
        self.W_f = nn.Parameter(torch.Tensor(input_size,
                                             hidden_size).uniform_(-0.1, 0.1))
        self.W_o = nn.Parameter(torch.Tensor(input_size,
                                             hidden_size).uniform_(-0.1, 0.1))
        self.W_u = nn.Parameter(torch.Tensor(input_size,
                                             hidden_size).uniform_(-0.1, 0.1))
        self.U_i = nn.Parameter(torch.Tensor(hidden_size,
                                             hidden_size).uniform_(-0.1, 0.1))
        self.U_f = nn.Parameter(torch.Tensor(hidden_size,
                                             hidden_size).uniform_(-0.1, 0.1))
        self.U_o = nn.Parameter(torch.Tensor(hidden_size,
                                             hidden_size).uniform_(-0.1, 0.1))
        self.U_u = nn.Parameter(torch.Tensor(hidden_size,
                                             hidden_size).uniform_(-0.1, 0.1))
        self.b_i = nn.Parameter(torch.Tensor(1, hidden_size))
        self.b_f = nn.Parameter(torch.Tensor(1, hidden_size))
        self.b_o = nn.Parameter(torch.Tensor(1, hidden_size))
        self.b_u = nn.Parameter(torch.Tensor(1, hidden_size))

    def forward(self, x, h, C):
        # Currently, batching (probably) does not work. Probably should be
        # fixed, at least for the sake of being consistent with the pytorch
        # recurrent models. x should be (N, D) where N is batch size and D
        # input size, h and C would be (N, C, H) where N is batch size, C is
        # the number of children and H is hidden size. Calculation of f_j
        # should be changed. Probabluy can be done with a nifty matrix op.
        h_tilde = torch.sum(h, 0).squeeze()
        i_j = torch.sigmoid(x @ self.W_i + h_tilde @ self.U_i + self.b_i)
        o_j = torch.sigmoid(x @ self.W_o + h_tilde @ self.U_o + self.b_o)
        f_x = x @ self.W_f
        f_j = torch.stack([torch.sigmoid(f_x + f @ self.U_f + self.b_f)
                           for f in h])
        u_j = torch.tanh(x @ self.W_u + h_tilde @ self.U_u + self.b_u)
        c_j = i_j * u_j + torch.sum(f_j * C, 0).squeeze()
        h_j = o_j * torch.tanh(c_j)
        return h_j, c_j


class BaselineNetwork(nn.Module):
    """A baseline network architecture for natural language inference. Each
    sentence is represented as a bag-of-words -- the sum of the word
    embeddings. This model is based on the bag-of-words baseline of Bowman et
    al. (2015).

    Args:
        embeddings (torch.LongTensor): A torch.LongTensor containing
        pre-trained word embeddings of dimensions (N, D) where N is the number
        of embeddings and D is the dimensionality of the embeddings.
    """
    def __init__(self, embeddings):
        super(BaselineNetwork, self).__init__()
        self.wemb = nn.Embedding(embeddings.size(0), embeddings.size(1))
        self.wemb.weight = nn.Parameter(embeddings)
        self.emb_transform = nn.Linear(embeddings.size(1), 100)
        self.fc1 = nn.Linear(200, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, 3)

    def forward(self, premise, hypothesis):
        x_premise = self.wemb(premise).sum(1).squeeze(1)
        x_hypothesis = self.wemb(hypothesis).sum(1).squeeze(1)
        x_premise = F.tanh(self.emb_transform(x_premise))
        x_hypothesis = F.tanh(self.emb_transform(x_hypothesis))
        x = torch.cat((x_premise, x_hypothesis), 1)
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = F.tanh(self.fc3(x))
        return F.log_softmax(x)


class SNLICorpus(torch.utils.data.Dataset):
    """Class for representing a specific split of the SNLI corpus.

    Args:
        path (string): Path to the corpus.
        vocab (dict): Dictionary mapping words to integers.
        tokenizer (tokenizer): A spaCy tokenizer used to tokenize the
        sentences.
        pad (bool, optional): Whether to pad sentences in order to allow for
        batching the input to the network.
        seq_length (int, optional): Max length of sentences. Sentences longer
        than this value will be cropped, and sentences that are shorter will be
        padded if pad is set to True.
    """
    def __init__(self, path, vocab, tokenizer, pad=False, seq_length=30):
        self.premises = []
        self.hypotheses = []
        self.labels = []
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.pad = pad
        self.seq_length = seq_length
        self.gold_labels = {'neutral': [0],
                            'entailment': [1],
                            'contradiction': [2]}
        self._load_data(path)

    def _load_data(self, path):
        with open(path, 'r') as f:
            for i, line in enumerate(f):
                instance = json.loads(line)
                label = instance.get('gold_label')
                if label == '-':
                    continue
                gold_label = self.gold_labels.get(label)
                premise = [self.vocab.get(x.text.lower(), 1) for x in
                           self.tokenizer(instance.get('sentence1'))]
                hypothesis = [self.vocab.get(x.text.lower(), 1) for x in
                              self.tokenizer(instance.get('sentence2'))]
                if self.pad:
                    if len(premise) > self.seq_length:
                        premise = premise[:self.seq_length]
                    else:
                        premise.extend([0 for _ in range(
                                        self.seq_length - len(premise))])
                    if len(hypothesis) > self.seq_length:
                        hypothesis = hypothesis[:self.seq_length]
                    else:
                        hypothesis.extend([0 for _ in range(
                                           self.seq_length - len(hypothesis))])
                self.premises.append(torch.LongTensor(premise))
                self.hypotheses.append(torch.LongTensor(hypothesis))
                self.labels.append(torch.LongTensor(gold_label))

    def __getitem__(self, idx):
        return self.premises[idx], self.hypotheses[idx], self.labels[idx]

    def __len__(self):
        return len(self.premises)


def load_embeddings(path):
    """Load pretrained word embeddings into format appropriate for the
    PyTorch network. Assumes word vectors are kept in a file with one vector
    per line, where the first column is the corresponding word.

    Args:
        path (str): Path to the word embeddings.
    Returns:
        dict: A dictionary mapping words to their vector index.
        torch.LongTensor: A torch.LongTensor of dimension (N, D) where N is
        the size of the vocabulary and D is the dimension of the embeddings.
    """
    embeddings = [[0 for _ in range(100)],
                  [random.uniform(-1, 1) for _ in range(100)]]
    word2id = {"<PAD>": 0}
    word2id = {"<UNKNOWN>": 1}
    with open(path, 'r') as f:
        for line in f:
            entry = line.strip().split()
            word = entry[0]
            word2id.update({word: len(word2id)})
            embeddings.append([float(x) for x in entry[1:]])

    return word2id, torch.FloatTensor(embeddings)


def load_data(path, word2id):
    """Load SNLI corpus (nlp.stanford.edu/projects/snli/) into a format that can
    be used for training and testing a network.

    Args:
        path (str): Path to the corpus.
        word2id (dict): A dictionary mapping words to integers.

    Returns:
        list: A list of tuples where each tuple (t, h, l) is a training/test
        instance and each tuple element is a torch.LongTensor representing the
        sentence or the class label. t is the premise, h is the hypothesis and
        l is the true label of the example.
    """
    labels = {'neutral': 0,
              'entailment': 1,
              'contradiction': 2}
    data = []
    with open(path, 'r') as f:
        for line in f:
            instance = json.loads(line)
            label = instance.get('gold_label')
            if label == '-':
                continue
            else:
                premise = [word2id.get(x.lower(), 1) for x in
                           instance.get('sentence1_binary_parse').split()
                           if x.isalnum()]
                hypothesis = [word2id.get(x.lower(), 1) for x in
                              instance.get('sentence2_binary_parse').split()
                              if x.isalnum()]
                data.append((torch.LongTensor([premise]),
                             torch.LongTensor([hypothesis]),
                             torch.LongTensor([labels.get(label)])))
    return data


def train(model, data_loader, optimizer, epoch):
    model.train()
    correct = 0
    for batch, (premise, hypothesis, target) in enumerate(data_loader):
        premise = Variable(premise)
        hypothesis = Variable(hypothesis)
        target = Variable(target.squeeze())
        optimizer.zero_grad()
        output = model(premise, hypothesis)
        loss = F.nll_loss(output, target)
        _, pred = output.data.max(1)
        correct += pred.eq(target.data).sum()
        loss.backward()
        optimizer.step()
        if batch % 50 == 0:
            print('Epoch {}, batch: {}/{} \tLoss: {:.6f}'.format(
                epoch, batch, len(data_loader.dataset) //
                data_loader.batch_size, loss.data[0]))

    print('Training accuracy epoch {}: {:d}/{:d} ({:.2%})'.format(epoch,
          correct, len(data_loader.dataset), correct / len(data_loader.dataset)
    ))


def test(model, data):
    model.eval()
    test_loss = 0
    correct = 0
    for batch, (premise, hypothesis, target) in enumerate(data):
        premise = Variable(premise, volatile=True)
        hypothesis = Variable(hypothesis, volatile=True)
        target = Variable(target.squeeze())
        output = model(premise, hypothesis)
        test_loss += F.cross_entropy(output, target).data[0]
        _, pred = output.data.max(1)
        correct += pred.eq(target.data).sum()

    print('\nTest:\nAverage loss: {:.4f}, Accuracy: {:d}/{:d} ({:.2%})\n'.format(
        test_loss / len(data), correct, len(data), correct / len(data)
    ))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--embeddings', type=str,
                        default='embeddings/glove.6B.100d.txt',
                        help='Path to pretrained word embeddings.')
    parser.add_argument('--train', type=str,
                        default='data/snli_1.0_train.jsonl',
                        help='Path to training data')
    parser.add_argument('--dev', type=str,
                        default='data/snli_1.0_dev.jsonl',
                        help='Path to development data.')
    parser.add_argument('--lr', type=float, default=0.2,
                        help='learning rate')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Size of mini-batches.')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs to train for.')
    parser.add_argument('--save', type=str, default='model.pt',
                        help='Path to save the trained model.')
    args = parser.parse_args()

    print("Loading data.")
    vocabulary, embeddings = load_embeddings(args.embeddings)
    tokenizer = spacy.load('en', tagger=False, entity=False, parser=False)
    train_loader = torch.utils.data.DataLoader(SNLICorpus(
        args.train, vocabulary, tokenizer, pad=True), batch_size=128,
        shuffle=True, num_workers=1)
    dev_loader = torch.utils.data.DataLoader(SNLICorpus(
        args.dev, vocabulary, tokenizer), batch_size=1)
    model = BaselineNetwork(embeddings)

    learning_rate = args.lr
    momentum = 0.5
    for epoch in range(1, args.epochs + 1):
        optimizer = optim.SGD(model.parameters(), lr=learning_rate,
                              momentum=momentum)
        train(model, train_loader, optimizer, epoch)
        test(model, dev_loader)
        momentum = momentum * 1.2 if momentum < 0.8 else 0.9
        # halve learning rate every other epoch
        if epoch % 2 == 0:
            learning_rate *= 0.5
    # with open(args.save, 'wb') as f:
    #     torch.save(model, f)
