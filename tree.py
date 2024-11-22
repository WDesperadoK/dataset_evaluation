# Import necessary libraries
from nltk import Tree
import zss

# Define a Node class that zss can use
class Node:
    def __init__(self, label):
        self.label = label
        self.children = []

    def add_child(self, child):
        self.children.append(child)

    def __repr__(self):  # This makes debugging easier
        return f"{self.label}({', '.join(repr(child) for child in self.children)})"

# Function to get children of a node (required by zss)
def get_children(node):
    return node.children

# Function to get label of a node
def get_label(node):
    return node.label

# Define the cost functions
def label_dist(label1, label2):
    """Define how costly it is to change label1 into label2."""
    return 0 if label1 == label2 else 1

def calculate_ted(tree1, tree2):
    # Calculate tree edit distance using defined functions
    return zss.simple_distance(tree1, tree2)

# Function to convert NLTK tree to custom Node format
def parse_corenlp_output_to_tree(corenlp_output):
    # Parse the string using NLTK's Tree
    tree = Tree.fromstring(corenlp_output)

    # Convert this NLTK tree to our Node format
    def convert(nltk_tree):
        if isinstance(nltk_tree, str):
            return Node(nltk_tree)
        node = Node(nltk_tree.label())
        for child in nltk_tree:
            node.add_child(convert(child))
        return node

    return convert(tree)

# TED 3
def parse_corenlp_output_to_tree_3(corenlp_output, max_depth=3):
    # Parse the string using NLTK's Tree
    tree = Tree.fromstring(corenlp_output)

    # Convert this NLTK tree to our Node format, limiting the depth
    def convert(nltk_tree, current_depth=0):
        if isinstance(nltk_tree, str) or current_depth >= max_depth:
            return Node(nltk_tree) if isinstance(nltk_tree, str) else Node(nltk_tree.label())
        
        node = Node(nltk_tree.label())
        for child in nltk_tree:
            if current_depth < max_depth:
                node.add_child(convert(child, current_depth + 1))
        return node

    return convert(tree)

# Simple distance function for zss
def simple_distance(node1, node2):
    """A simple distance function comparing node labels."""
    return 1 if node1.label != node2.label else 0


# Main function to handle the workflow
def main():
    # Example trees from Stanford CoreNLP output
    tree1_str = "(ROOT (FRAG (FRAG (ADVP (RB aside)) (NP (NP (JJ general) (NN consensus)) (PP (IN within) (NP (NN namlp) (NN community))))) (FRAG (NP (NN stratification)) (ADJP (JJ essential)))))"
    tree2_str = "(ROOT (S (NP (NP (JJ collective) (NN viewpoint)) (PP (IN within) (NP (NN namlp) (NN community) (NN stratification)))) (ADJP (JJ vital))))"

    # Convert raw tree strings to Node format
    tree1 = parse_corenlp_output_to_tree(tree1_str)
    tree2 = parse_corenlp_output_to_tree(tree2_str)

    ted3_tree1 = parse_corenlp_output_to_tree_3(tree1_str)
    ted3_tree2 = parse_corenlp_output_to_tree_3(tree2_str)

    # Calculate and print the tree edit distance
    distance = calculate_ted(tree1, tree2)
    print("Tree Edit Distance:", distance)

    # Calculate and print the tree edit distance with max depth of 3
    distance = calculate_ted(ted3_tree1, ted3_tree2)
    print("Tree Edit Distance (Max Depth 3):", distance)

# Check if the script is run as the main program
if __name__ == "__main__":
    main()
