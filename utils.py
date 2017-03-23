import StanfordDependencies
import json
import torch


class Node(object):
    def __init__(self, idx, word, head):
        self.idx = idx
        self.word = word
        self.head = head
        self.children = set()


class Tree(object):
    def __init__(self):
        self.nodes = []


def get_queue_item(lst, idx):
    try:
        return lst[idx]
    except IndexError:
        return 0


def get_tree(sentence):
    dep_tree = Tree()
    for word in sentence:
        fields = word.split(":")
        node = Node(fields[0], fields[1], fields[2])
        dep_tree.nodes.append(node)
    for node in dep_tree.nodes:
        dep_tree.nodes[node.head].children.add(node.idx)
    return dep_tree


def convert_trees(input_file, output_file):
    sd = StanfordDependencies.get_instance(backend='jpype')
    with open(input_file, 'r') as inpf, open(output_file, 'w') as outf:
        for line in inpf:
            sentence = json.loads(line)
            prem = sentence.get('sentence1_parse')
            hypo = sentence.get('sentence2_parse')
            prem_dep = ' '.join(['{}:{}:{}'.format(x.index, x.form, x.head)
                                 for x in sd.convert_tree(prem)])
            hypo_dep = ' '.join(['{}:{}:{}'.format(x.index, x.form, x.head)
                                 for x in sd.convert_tree(hypo)])
            sentence.update({'sentence1_dependency_parse': prem_dep,
                             'sentence2_dependency_parse': hypo_dep})
            outf.write(json.dumps(sentence) + "\n")

    print("Wrote file with dependency parse annotation to {}".format(
           output_file))


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
