import StanfordDependencies
import json
import copy
import argparse


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
                        default='data/glove100d-cache.pt',
                        help='Path from where to load or save vector cache')
    parser.add_argument('--vocab', type=str, default='data/vocabulary.pkl')
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
    parser.add_argument('--epochs', type=int, default=10,
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
