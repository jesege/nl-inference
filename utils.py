import StanfordDependencies
import json


class DefaultList(list):
    """A list that returns a default value if index out of bounds."""
    def __init__(self, default=None):
        self.default = default
        list.__init__(self)

    def __getitem__(self, index):
        try:
            return list.__getitem__(self, index)
        except IndexError:
            return self.default


class Node(object):
    def __init__(self, idx, word, head):
        self.idx = idx
        self.word = word
        self.head = head
        self.children = set()


class Tree(object):
    def __init__(self):
        self.nodes = []


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
