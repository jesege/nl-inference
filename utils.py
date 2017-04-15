import StanfordDependencies
import json
import copy


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
