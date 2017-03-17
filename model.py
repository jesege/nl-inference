import argparse
import random
import json
import math
import spacy
import utils
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable


class SPINNEncoder(nn.Module):
    """Implementation of the SPINN architecture.
    """
    def __init__(self, embeddings, hidden_size, tracking_lstm=None):
        super(SPINNEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.wemb = nn.Embedding(embeddings.size(0), embeddings.size(1))
        self.wemb.weight = embeddings
        self.wemb.requires_grad = False
        self.embedding_transform = nn.Linear(embeddings.size(1), hidden_size)
        if tracking_lstm is None:
            self.tracking_lstm = None
            self.x_size = hidden_size
        else:
            self.tracking_lstm = nn.LSTMCell(3 * hidden_size, 64)
        self.enc = BinaryTreeLSTMCell(self.x_size, self.hidden_size)

    def forward(self, sequence, transitions):
        # Input N x L where N is size of mini-batch and L is length of seq.
        # Goes for both sequence and transitions
        # Maybe is a good idea to separate the embedding transformation
        # and get the sequence input as a matrix of embeddings instead.
        batch_size = sequence.size(0)
        timesteps = transitions.size(1)
        stack = torch.zeros((batch_size, transitions.size(1) + 1,
                             self.hidden_size * 2))
        buffer = self.embedding_transform(self.wemb(sequence)).repeat(1, 1, 2)
        backpointerq = [utils.DefaultList(0) for _ in range(batch_size)]
        buffer_ptr = torch.zeros(batch_size)
        # if we transpose the transitions matrix we can index it like
        # transitions[timestep] to get all the data for the timestep
        transitions = transitions.t()
        for timestep in range(1, timesteps + 1):
            mask = transitions[timestep - 1]
            stk_ptr_1 = torch.LongTensor([backpointerq[i][-1]
                                          for i in range(len(backpointerq))])
            stk_ptr_2 = torch.LongTensor([backpointerq[i][-2]
                                          for i in range(len(backpointerq))])
            buffer_top = buffer[buffer_ptr]
            # New stack tops if transition would be reduce for all examples
            right = torch.Tensor(batch_size, self.hidden_size * 2)
            left = torch.Tensor(batch_size, self.hidden_size * 2)
            for i, p1, p2 in zip(range(batch_size), stk_ptr_1, stk_ptr_2):
                right[i] = stack[i][p1]
                left[i] = stack[i][p2]

            if self.tracking_lstm is None:
                x = torch.zeros(self.x_size)
            else:
                # TODO: get output from t-lstm
                pass
            stack_tops = torch.cat(self.enc(x, right, left), 1)
            # Update the stack with new values. Here we might as well do a
            # a branching since we loop over each example anyway?
            for i in range(batch_size):
                if mask[i] == 1:
                    stack[i][timestep] = stack_tops[i]
                else:
                    stack[i][timestep] = buffer_top[i]

            # Move the buffer and stack pointers
            buffer_ptr = buffer_ptr + (1 - mask)
            for i in range(len(backpointerq)):
                if mask[i] == 1:
                    # We are probably fine not safekeeping against lists
                    # shorter than 2 elements, since if they appear here,
                    # we have bigger problems and we can view this laziness
                    # as a debugging feature.
                    backpointerq[i] = backpointerq[i][:-2]
                backpointerq[i].append(timestep)

        # return the elements from the last timestep
        return stack.transpose(0, 1)[-1]


class CSTreeLSTMCell(nn.Module):
    """A Child-Sum Tree-Long Short Term Memory cell.

    """
    def __init__(self, input_size, hidden_size):
        super(CSTreeLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.W_i = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.W_f = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.W_o = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.W_u = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.U_i = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.U_f = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.U_o = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.U_u = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.reset_parameters()
        # Initialize biases after resetting of parameters; they can be zeroes.
        self.b_i = nn.Parameter(torch.Tensor(1, hidden_size))
        self.b_f = nn.Parameter(torch.Tensor(1, hidden_size))
        self.b_o = nn.Parameter(torch.Tensor(1, hidden_size))
        self.b_u = nn.Parameter(torch.Tensor(1, hidden_size))

    def reset_parameters(self):
        stdev = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdev, stdev)

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


class BinaryTreeLSTMCell(nn.Module):
    """ An Binary Tree-Long-Short Term Memory cell as defined by
    Tai et al. (2015).
    """
    def __init__(self, input_size, hidden_size):
        super(BinaryTreeLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.W_i = nn.Linear(input_size, hidden_size, bias=False)
        self.W_f = nn.Linear(input_size, hidden_size, bias=False)
        self.W_o = nn.Linear(input_size, hidden_size, bias=False)
        self.W_u = nn.Linear(input_size, hidden_size, bias=False)
        self.U_il = nn.Linear(hidden_size, hidden_size, bias=False)
        self.U_ir = nn.Linear(hidden_size, hidden_size, bias=False)
        self.U_fl = nn.Linear(hidden_size, hidden_size, bias=False)
        self.U_fr = nn.Linear(hidden_size, hidden_size, bias=False)
        self.U_ol = nn.Linear(hidden_size, hidden_size, bias=False)
        self.U_or = nn.Linear(hidden_size, hidden_size, bias=False)
        self.U_ur = nn.Linear(hidden_size, hidden_size, bias=False)
        self.U_ul = nn.Linear(hidden_size, hidden_size, bias=False)
        # self.W_i = nn.Parameter(torch.Tensor(input_size, hidden_size))
        # self.W_f = nn.Parameter(torch.Tensor(input_size, hidden_size))
        # self.W_o = nn.Parameter(torch.Tensor(input_size, hidden_size))
        # self.W_u = nn.Parameter(torch.Tensor(input_size, hidden_size))
        # self.U_il = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        # self.U_ir = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        # self.U_fl = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        # self.U_fr = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        # self.U_ol = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        # self.U_or = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        # self.U_ur = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        # self.U_ul = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.init_parameters()
        # Initialize biases after resetting of parameters; they can be zeroes.
        self.b_i = nn.Parameter(torch.Tensor(1, hidden_size))
        self.b_f = nn.Parameter(torch.Tensor(1, hidden_size))
        self.b_o = nn.Parameter(torch.Tensor(1, hidden_size))
        self.b_u = nn.Parameter(torch.Tensor(1, hidden_size))

    def init_parameters(self):
        for weight in self.parameters():
            init.kaiming_uniform(weight.data)

    def forward(self, x, left, right):
        # Inputs = (B X D)
        # The inputting is a bit unituitive and tailor made to the SPINN
        # encoder. Might be bad. But it works and we are probably not gonna
        # use it for anything else.
        h_left, c_left = left.split(self.hidden_size, dim=1)
        h_right, c_right = right.split(self.hidden_size, dim=1)
        i_j = torch.sigmoid(self.W_i(x) + torch.sum(torch.stack(
            (self.U_il(h_left), self.U_ir(h_right))), 0).squeeze() +
             self.b_i.expand(x.size(0), self.hidden_size))
        o_j = torch.sigmoid(self.W_o(x) + torch.sum(torch.stack(
            (self.U_ol(h_left), self.U_or(h_right))), 0).squeeze() +
             self.b_o.expand(x.size(0), self.hidden_size))
        f_jl = torch.sigmoid(self.W_o(x) + torch.sum(torch.stack(
            (self.U_fl(h_left), self.U_fr(h_right))), 0).squeeze() +
             self.b_f.expand(x.size(0), self.hidden_size))
        f_jr = torch.sigmoid(self.W_o(x) + torch.sum(torch.stack(
            (self.U_fl(h_left), self.U_fr(h_right))), 0).squeeze() +
             self.b_f.expand(x.size(0), self.hidden_size))
        u_j = torch.tanh(self.W_u(x) + torch.sum(torch.stack(
            (self.U_ul(h_left), self.U_ur(h_right))), 0).squeeze() +
             self.b_u.expand(x.size(0), self.hidden_size))
        c_j = i_j * u_j + torch.sum(torch.stack((
            f_jl * c_left, f_jr * c_right)), 0).squeeze()
        h_j = o_j * torch.tanh(c_j)
        # i_j = torch.sigmoid(x @ self.W_i + torch.sum(torch.cat(
        #     (h_left @ self.U_il, h_right @ self.U_ir)), 0) + self.b_i)
        # o_j = torch.sigmoid(x @ self.W_o + torch.sum(torch.cat(
        #     (h_left @ self.U_ol, h_right @ self.U_or)), 0) + self.b_o)
        # f_jl = torch.sigmoid(x @ self.W_o + torch.sum(torch.cat(
        #     (h_left @ self.U_fl, h_right @ self.U_fr)), 0) + self.b_f)
        # f_jr = torch.sigmoid(x @ self.W_o + torch.sum(torch.cat(
        #     (h_left @ self.U_fl, h_right @ self.U_fr)), 0) + self.b_f)
        # u_j = torch.tanh(x @ self.W_u + torch.sum(torch.cat(
        #     (h_left @ self.U_ul, h_right @ self.U_ur)), 0) + self.b_u)
        # c_j = i_j * u_j + torch.sum(torch.cat((
        #     f_jl * c_left, f_jr * c_right)), 0).squeeze()
        # h_j = o_j * torch.tanh(c_j)
        return h_j, c_j


class MLPClassifier(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(input_size, 3)

    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        return F.log_softmax(x)


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
        self.binary_premise_parse = []
        self.binary_hypothesis_parse = []
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
                self.binary_premise_parse.append(get_constituency_transitions(
                    instance.get('sentence1_binary_parse')
                ))
                self.binary_hypothesis_parse.append(get_constituency_transitions(
                    instance.get('sentence2_binary_parse')
                ))

    def __getitem__(self, idx):
        return self.premises[idx], self.hypotheses[idx], self.labels[idx]

    def __len__(self):
        return len(self.premises)


def get_constituency_transitions(sentence):
    """Return the transitions that lead to the given binary parse tree.

    Args:
        sentence (str): The sentece represented as a binary parse in the form
        of an s-expression.
    Returns:
        list: A list of transitions in {"SHIFT", "REDUCE"} that leads to this
        parse tree.
    """
    transitions = []
    for token in sentence.split():
        if token == "(":
            continue
        elif token == ")":
            transitions.append("REDUCE")
        else:
            transitions.append("SHIFT")

    return transitions


def get_dependency_transitions(sentence):
    """Return a list of transitions that lead to the given dependency tree.

    Args:
        sentence (list): A list of tokens in the sentence where each element
        is structured as "idx(int):word(str):head(int)".
    Returns:
        list: A list of transitions in {"LA", "RA", "SHIFT"} that leads to
        this parse tree.
    """
    gold_tree = utils.get_tree(sentence)
    buffer = [x.split(":")[0] for x in sentence]
    stack = []
    transitions = []
    while buffer and stack:
        if not stack:
            transitions.append("sh")


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
                  [random.uniform(-0.1, 0.1) for _ in range(100)]]
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

    with open(args.save, 'wb') as f:
        torch.save(model, f)
