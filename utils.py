import re
import json
import copy
import argparse
import matplotlib.pyplot as plt


class Node(object):
    def __init__(self, idx, head, word=None):
        self.idx = idx
        self.head = head
        self.word = word
        self.children = set()


class Tree(object):
    def __init__(self):
        self.nodes = []
        self.node_idxs = []

    def update(self, head, child):
        self.nodes[child].head = head
        self.nodes[head].children.add(child)

    def add_node(self, node):
        self.nodes.append(node)

    def __getitem__(self, idx):
        return self.nodes[idx]

    def __iter__(self):
        for node in self.nodes:
            yield node


def get_parse_tree(gold_tree):
    parse_tree = copy.deepcopy(gold_tree)
    for node in parse_tree.nodes:
        node.head = None
        node.children = set()
    return parse_tree


def get_gold_tree(sentence):
    dep_tree = Tree()
    root = Node(0, None, word='ROOT')
    dep_tree.nodes.append(root)
    for word in sentence.split(' '):
        idx, token, head, deprel = word.split("(")
        node = Node(int(idx), int(head), word=token)
        dep_tree.add_node(node)
    for node in dep_tree:
        if node.idx == 0:
            continue
        head = dep_tree[node.head]
        dep_tree[node.head].children.add(node.idx)
    return dep_tree


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--embeddings', type=str,
                        default='embeddings/glove.6B.100d.txt',
                        help='Path to pretrained word embeddings.')
    parser.add_argument('--wv-cache', type=str,
                        help='Path from where to load or save vector cache')
    parser.add_argument('--training-cache', type=str,
                        help='''Path to save or load a serialized version of
                        training corpus. HIGHLY recommended for dependency
                        version.''')
    parser.add_argument('--vocab', type=str)
    parser.add_argument('--wdim', type=int, default=100,
                        help='Dimensionality of the provided embeddings.')
    parser.add_argument('--edim', type=int, default=100,
                        help='Dimensionality of the sentence encoder.')
    parser.add_argument('--tdim', type=int, default=64,
                        help='Dimensionality of the tracking LSTM.')
    parser.add_argument('--tracking', action='store_true',
                        help='Use a tracking LSTM.')
    parser.add_argument('--dependency', action='store_true',
                        help='Use the dependency based encoder.')
    parser.add_argument('--bow', action='store_true',
                        help='Use a bag-of-words encoder.')
    parser.add_argument('--train', type=str,
                        default='data/snli_1.0_train.jsonl',
                        help='Path to training data.')
    parser.add_argument('--dev', type=str,
                        default='data/snli_1.0_dev.jsonl',
                        help='Path to development data.')
    parser.add_argument('--lr', type=float, default=2e-3,
                        help='Learning rate')
    parser.add_argument('--l2', type=float, default=3e-5,
                        help='L2 regularization term.')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Size of mini-batches.')
    parser.add_argument('--epochs', type=int, default=15,
                        help='Number of epochs to train for.')
    parser.add_argument('--log-interval', type=int, default=100,
                        help='Logs every n batches.')
    parser.add_argument('--log-path', type=str, default='model.log',
                        help='File to write training log to.')
    parser.add_argument('--lr-decay-every', type=int, default=10000,
                        help='Decay lr every n steps.')
    parser.add_argument('--lr-decay', type=float, default=0.75,
                        help='Learning rate decay factor.')
    parser.add_argument('--save', type=str, default='models/model',
                        help='Path to save the trained model.')
    parser.add_argument('--model', type=str, help='Path to a saved model.')
    parser.add_argument('--test', type=str, help="""Path to testing portion of
                        the corpus. If provided, --model argument has to be
                        given too.""")
    return parser.parse_args()


def convert_trees(input_file, output_file):
    import StanfordDependencies
    sd = StanfordDependencies.get_instance(backend='jpype')
    with open(input_file, 'r') as inpf, open(output_file, 'w') as outf:
        for line_no, line in enumerate(inpf):
            if line_no % 1000 == 0:
                print("Processing sentence pair {}.".format(line_no))
            sentence = json.loads(line)
            prem = sentence.get('sentence1_parse')
            hypo = sentence.get('sentence2_parse')
            prem_dep = ' '.join(['{}({}({}({}'.format(x.index, x.form, x.head, x.deprel)
                                 for x in sd.convert_tree(prem)])
            hypo_dep = ' '.join(['{}({}({}({}'.format(x.index, x.form, x.head, x.deprel)
                                 for x in sd.convert_tree(hypo)])
            sentence.update({'sentence1_dependency_parse': prem_dep,
                             'sentence2_dependency_parse': hypo_dep})
            outf.write(json.dumps(sentence) + "\n")

    print("Wrote file with dependency parse annotation to {}".format(
           output_file))


def plot_training(training_acc, validation_acc):
    plt.plot(training_acc)
    plt.plot(validation_acc)
    plt.title("Model accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Iteration")
    plt.legend(["train", "dev"], loc='upper left')
    plt.show()


def parse_log(log_file):
    iteration = []
    accuracy = []
    accuracy_pattern = re.compile(r"0\.\d{4}")
    with open(log_file, 'r') as f:
        for line in f:
            if 'Iteration' in line:
                fields = line.split(' ')
                for field in fields:
                    if field.endswith("0:"):
                        iteration.append(int(field[:-1]))
                    elif field.endswith(","):
                        accuracy.append(int(field[:-1]))
                    elif accuracy_pattern.match(field):
                        accuracy.append(field)
    it_acc = []
    for it, acc in zip(accuracy, iteration):
        it_acc.append([it, acc])
    return it_acc

"""
2017-04-18 19:41 - model.training - INFO - Iteration 5000: dev acc. 0.6771, dev loss 0.74925
2017-04-18 20:14 - model.training - INFO - Iteration 10000: dev acc. 0.6909, dev loss 0.71023
2017-04-18 20:47 - model.training - INFO - Iteration 15000: dev acc. 0.7320, dev loss 0.64877
2017-04-18 21:20 - model.training - INFO - Iteration 20000: dev acc. 0.7143, dev loss 0.66592
2017-04-18 21:53 - model.training - INFO - Iteration 25000: dev acc. 0.7373, dev loss 0.63299
2017-04-18 22:25 - model.training - INFO - Iteration 30000: dev acc. 0.7549, dev loss 0.60655
2017-04-18 22:58 - model.training - INFO - Iteration 35000: dev acc. 0.7487, dev loss 0.61934
2017-04-18 23:31 - model.training - INFO - Iteration 40000: dev acc. 0.7568, dev loss 0.60525
2017-04-19 00:04 - model.training - INFO - Iteration 45000: dev acc. 0.7647, dev loss 0.60175
2017-04-19 00:36 - model.training - INFO - Iteration 50000: dev acc. 0.7469, dev loss 0.61086
2017-04-19 01:09 - model.training - INFO - Iteration 55000: dev acc. 0.7393, dev loss 0.62512
2017-04-19 01:42 - model.training - INFO - Iteration 60000: dev acc. 0.7506, dev loss 0.61319
2017-04-19 02:15 - model.training - INFO - Iteration 65000: dev acc. 0.7508, dev loss 0.61760
2017-04-19 02:47 - model.training - INFO - Iteration 70000: dev acc. 0.7445, dev loss 0.63020
2017-04-19 03:20 - model.training - INFO - Iteration 75000: dev acc. 0.7398, dev loss 0.64129
2017-04-19 03:53 - model.training - INFO - Iteration 80000: dev acc. 0.7419, dev loss 0.63552
2017-04-19 04:26 - model.training - INFO - Iteration 85000: dev acc. 0.7387, dev loss 0.65626
2017-04-19 04:58 - model.training - INFO - Iteration 90000: dev acc. 0.7444, dev loss 0.64445
2017-04-19 05:31 - model.training - INFO - Iteration 95000: dev acc. 0.7359, dev loss 0.66141
2017-04-19 06:04 - model.training - INFO - Iteration 100000: dev acc. 0.7320, dev loss 0.68129
2017-04-19 06:37 - model.training - INFO - Iteration 105000: dev acc. 0.7251, dev loss 0.70015
2017-04-19 07:10 - model.training - INFO - Iteration 110000: dev acc. 0.7313, dev loss 0.69362
2017-04-19 07:42 - model.training - INFO - Iteration 115000: dev acc. 0.7246, dev loss 0.71520
2017-04-19 08:15 - model.training - INFO - Iteration 120000: dev acc. 0.7293, dev loss 0.70641

2017-04-19 13:00 - model.training - INFO - Dev acc. iteration 10000: 0.6884 (loss: 0.74289)
2017-04-19 13:33 - model.training - INFO - Dev acc. iteration 15000: 0.7112 (loss: 0.69164)
2017-04-19 14:06 - model.training - INFO - Dev acc. iteration 20000: 0.7067 (loss: 0.69158)
2017-04-19 14:39 - model.training - INFO - Dev acc. iteration 25000: 0.7316 (loss: 0.66070)
2017-04-19 15:12 - model.training - INFO - Dev acc. iteration 30000: 0.7157 (loss: 0.67835)
2017-04-19 15:44 - model.training - INFO - Dev acc. iteration 35000: 0.7293 (loss: 0.65140)
2017-04-19 16:17 - model.training - INFO - Dev acc. iteration 40000: 0.7294 (loss: 0.64933)
2017-04-19 16:50 - model.training - INFO - Dev acc. iteration 45000: 0.7280 (loss: 0.66805)
2017-04-19 17:23 - model.training - INFO - Dev acc. iteration 50000: 0.7236 (loss: 0.66651)
2017-04-19 17:56 - model.training - INFO - Dev acc. iteration 55000: 0.7211 (loss: 0.67367)
2017-04-19 18:29 - model.training - INFO - Dev acc. iteration 60000: 0.7178 (loss: 0.69179)
2017-04-19 19:01 - model.training - INFO - Dev acc. iteration 65000: 0.7200 (loss: 0.67886)
2017-04-19 19:34 - model.training - INFO - Dev acc. iteration 70000: 0.7114 (loss: 0.72385)
2017-04-19 20:07 - model.training - INFO - Dev acc. iteration 75000: 0.7138 (loss: 0.71216)
2017-04-19 20:40 - model.training - INFO - Dev acc. iteration 80000: 0.7065 (loss: 0.75120)
2017-04-19 21:12 - model.training - INFO - Dev acc. iteration 85000: 0.7042 (loss: 0.75960)
2017-04-19 21:45 - model.training - INFO - Dev acc. iteration 90000: 0.6979 (loss: 0.77440)
2017-04-19 22:18 - model.training - INFO - Dev acc. iteration 95000: 0.6961 (loss: 0.79694)
2017-04-19 22:51 - model.training - INFO - Dev acc. iteration 100000: 0.7057 (loss: 0.77538)
2017-04-19 23:24 - model.training - INFO - Dev acc. iteration 105000: 0.6845 (loss: 0.86895)
2017-04-19 23:57 - model.training - INFO - Dev acc. iteration 110000: 0.6992 (loss: 0.81357)
2017-04-20 00:30 - model.training - INFO - Dev acc. iteration 115000: 0.6737 (loss: 0.93693)
2017-04-20 01:02 - model.training - INFO - Dev acc. iteration 120000: 0.6718 (loss: 0.96661)

"""