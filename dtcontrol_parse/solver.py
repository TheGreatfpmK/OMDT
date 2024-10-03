import numpy as np

from omdt.mdp import MarkovDecisionProcess
from omdt.tree import Tree, TreeLeaf, TreeNode

from .dtcontrol.decision_tree.decision_tree import DecisionTree
from .dtcontrol.decision_tree.impurity.entropy import Entropy
from .dtcontrol.decision_tree.splitting.axis_aligned import AxisAlignedSplittingStrategy, AxisAlignedSplit

import json
import sys

import time
from datetime import datetime

from dtcontrol.decision_tree.decision_tree import Node


def _make_set(v):
    if v is None:
        return set()
    if isinstance(v, tuple):
        return {v}
    try:
        return set(v)
    except TypeError:
        return {v}

def _get_unique_labels_from_2d(labels):
    """
    Computes unique labels of a 2d label array by mapping every unique inner array to an int. Returns the unique labels
    and the int mapping.
    """
    l = []
    int_to_label = {}
    next_unused_int = 1  # OC1 expects labels starting with 1
    label_str_to_int = {}
    for i in range(len(labels)):
        label_str = ','.join(sorted([str(i) for i in labels[i] if i != -1]))
        if label_str not in label_str_to_int:
            label_str_to_int[label_str] = next_unused_int
            int_to_label[next_unused_int] = labels[i]
            next_unused_int += 1
        new_label = label_str_to_int[label_str]
        l.append(new_label)
    return np.array(l), int_to_label

class SimpleDataset:
    """
    A simple dataset class that is limited to numerical observations and
    deterministic actions.
    """
    def __init__(self, observations: np.ndarray, predicted_actions: np.ndarray, name="unknown"):
        # We only allow numerical observations
        self.x = observations

        # We only allow a deterministic action in each state so axis 1 is size 1
        self.y = predicted_actions
        self.y = predicted_actions.reshape(-1, 1)

        self.name = name

        self.x_metadata = {"variables": None, "categorical": None, "category_names": None,
                           "min": None, "max": None, "step_size": None}
        self.y_metadata = {"categorical": [], "category_names": None, "min": None, "max": None, "step_size": None,
                           'num_rows': None, 'num_flattened': None}

        self.unique_labels_ = None

    def get_name(self):
        return self.name

    def is_deterministic(self):
        return True

    def get_single_labels(self):
        return self.y

    def get_unique_labels(self):
        """
        e.g.
        [[1  2  3 -1 -1],
         [1 -1 -1 -1 -1],
         [1  2 -1 -1 -1],
        ]

        gets mapped to

        unique_labels = [1, 2, 3]
        unique_mapping = {1: [1 2 3 -1 -1], 2: [1 -1 -1 -1 -1], 3: [1 2 -1 -1 -1]}
        """
        if self.unique_labels_ is None:
            self.unique_labels_, _ = _get_unique_labels_from_2d(self.y)
        return self.unique_labels_

    def get_numeric_x(self):
        return self.x

    def map_numeric_feature_back(self, feature):
        return feature

    def map_single_label_back(self, single_label):
        return single_label

    def index_label_to_actual(self, index_label):
        return index_label

    def compute_accuracy(self, y_pred):
        num_correct = 0
        for i in range(len(y_pred)):
            pred = y_pred[i]
            if pred is None:
                return None
            if set.issubset(_make_set(pred), set(self.y[i])):
                num_correct += 1
        return num_correct / len(y_pred)

    def from_mask_optimized(self, mask):
        empty_object = type('', (), {})()
        empty_object.parent_mask = mask
        empty_object.get_single_labels = lambda: self.y[mask]
        return empty_object

    def from_mask(self, mask):
        subset = SimpleDataset(self.x[mask], self.y[mask], self.name)
        subset.parent_mask = mask

        if self.unique_labels_ is not None:
            subset.unique_labels_ = self.unique_labels_[mask]
        return subset

    def load_metadata_from_json(self, json_object):
        metadata = json_object['metadata']
        self.x_metadata = metadata['X_metadata']
        self.y_metadata = metadata['Y_metadata']

    def __len__(self):
        return len(self.x)

    def load_if_necessary(self):
        pass

    def set_treat_categorical_as_numeric(self):
        pass

def _dtcontrol_node_to_omdt_rec(node):
    if node.is_leaf():
        return TreeLeaf(node.actual_label)

    assert len(node.children) == 2

    left_child = _dtcontrol_node_to_omdt_rec(node.children[0])
    right_child = _dtcontrol_node_to_omdt_rec(node.children[1])
    return TreeNode(node.split.feature, node.split.threshold, left_child, right_child)


def _dtcontrol_tree_to_omdt(tree: DecisionTree):
    return Tree(_dtcontrol_node_to_omdt_rec(tree.root))

class DtControlSolverParser:
    def __init__(self, output_dir, verbose=False, scheduler_name="scheduler"):
        self.output_dir = output_dir
        self.verbose = verbose
        self.scheduler_name = scheduler_name

        # dtcontrol does not prove optimality
        self.optimal_ = False
        self.bound_ = 0

    def from_json_dict(self, json_dict, mdp, splitting_strategies=[AxisAlignedSplittingStrategy()], impurity_measure=Entropy(), depth=0):
        node = Node(splitting_strategies, impurity_measure, depth=depth)
        if json_dict["actual_label"] is not None:
            action = json_dict["actual_label"][0][:-1]
            try:
                node.actual_label = mdp.action_names.index(action)
            except:
                action = json_dict["actual_label"][0][:-2]
                node.actual_label = mdp.action_names.index(action)
            # node.index_label = mdp.action_names.index(action) # Assuming index_label is same as actual_label for simplicity
        if json_dict["split"] is not None:
            node.split = AxisAlignedSplit(mdp.feature_names.index(json_dict["split"]["lhs"]["var"]), json_dict["split"]["rhs"])
            for child_dict in json_dict["children"]:
                child_node = self.from_json_dict(child_dict, mdp, splitting_strategies, impurity_measure, depth + 1)
                node.children.append(child_node)
        node.num_nodes = 1 + sum(child.num_nodes for child in node.children)
        node.num_inner_nodes = 1 + sum(child.num_inner_nodes for child in node.children)
        return node

    def load_node_from_json(self, file_path, mdp):
        with open(file_path, 'r') as file:
            data = json.load(file)
        node = self.from_json_dict(data, mdp)
        return node
        
    def solve(self, mdp: MarkovDecisionProcess):
        # TODO fix scheduler name
        json_file_path = f'{mdp.path}{mdp.name}/decision_trees/default/{self.scheduler_name}/default.json'
        node = self.load_node_from_json(json_file_path, mdp)
        stats_json_file_path = f'{mdp.path}{mdp.name}/benchmark.json'
        with open(stats_json_file_path, 'r') as file:
            stats_data = json.load(file)
            parsed_time = datetime.strptime(stats_data[f'{self.scheduler_name}']['classifiers']['default']['time'], "%H:%M:%S.%f")
            parsed_time = parsed_time.replace(year=1970, hour=1)
            self.runtime = parsed_time.timestamp()
        self.tree_policy_ = Tree(_dtcontrol_node_to_omdt_rec(node))

    def act(self, obs):
        return self.tree_policy_.act(obs)

    def to_graphviz(
        self,
        feature_names,
        action_names,
        integer_features,
        colors=None,
        fontname="helvetica",
    ):
        return self.tree_policy_.to_graphviz(
            feature_names,
            action_names,
            integer_features,
            colors,
            fontname,
        )


