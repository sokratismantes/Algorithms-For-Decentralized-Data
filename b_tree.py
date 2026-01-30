import math
import hashlib

class BPlusTreeNode:
    def __init__(self, size):
        self.is_leaf = False
        self.values = []      # keys
        self.keys = []        # leaf: record lists ; internal: child pointers
        self.parent = None
        self.Next = None
        self.size = size      # max number of keys

    def insert_into_leaf(self, key, value):
        # key = record payload, value = sortable key
        if len(self.values) == 0:
            self.values = [value]
            self.keys = [[key]]
            return

        for i, val in enumerate(self.values):
            if value == val:
                self.keys[i].append(key)
                return
            elif value < val:
                self.values.insert(i, value)
                self.keys.insert(i, [key])
                return

        # insert at end
        self.values.append(value)
        self.keys.append([key])


class BPlusTree:
    def __init__(self, size):
        self.root = BPlusTreeNode(size)
        self.root.is_leaf = True

    def search(self, value):
        curr = self.root
        while not curr.is_leaf:
            i = 0
            while i < len(curr.values) and value >= curr.values[i]:
                i += 1
            curr = curr.keys[i]
        return curr
     
    def insert(self, key, value):
        leaf = self.search(value)
        leaf.insert_into_leaf(key, value)

        if len(leaf.values) > leaf.size:
            self.split_leaf(leaf)

    def split_leaf(self, leaf):
        mid = (leaf.size + 1) // 2

        new_leaf = BPlusTreeNode(leaf.size)
        new_leaf.is_leaf = True

        new_leaf.values = leaf.values[mid:]
        new_leaf.keys = leaf.keys[mid:]

        leaf.values = leaf.values[:mid]
        leaf.keys = leaf.keys[:mid]

        new_leaf.Next = leaf.Next
        leaf.Next = new_leaf

        new_leaf.parent = leaf.parent

        promoted_key = new_leaf.values[0]
        self.insert_in_parent(leaf, promoted_key, new_leaf)

    def insert_in_parent(self, node, value, new_node):
        if self.root == node:
            new_root = BPlusTreeNode(node.size)
            new_root.values = [value]
            new_root.keys = [node, new_node]
            new_root.is_leaf = False
            self.root = new_root
            node.parent = new_root
            new_node.parent = new_root
            return

        parent = node.parent

        idx = parent.keys.index(node)
        parent.values.insert(idx, value)
        parent.keys.insert(idx + 1, new_node)
        new_node.parent = parent

        if len(parent.values) > parent.size:
            self.split_internal(parent)

    def split_internal(self, node):
        mid = len(node.values) // 2
        promoted = node.values[mid]

        new_internal = BPlusTreeNode(node.size)
        new_internal.is_leaf = False

        new_internal.values = node.values[mid + 1:]
        new_internal.keys = node.keys[mid + 1:]

        for child in new_internal.keys:
            child.parent = new_internal

        node.values = node.values[:mid]
        node.keys = node.keys[:mid + 1]

        new_internal.parent = node.parent

        self.insert_in_parent(node, promoted, new_internal)

    # -------------------------
    # SEARCHING BY TITLE (SHA-1)
    # -------------------------
    def search_title(self, title):
        sha_key = int(hashlib.sha1(title.encode('utf-8')).hexdigest(), 16)
        return self.search_key(sha_key)

    def search_key(self, sha_key):
        leaf = self.search(sha_key)

        for i, v in enumerate(leaf.values):
            if v == sha_key:
                return leaf.keys[i]   # return list of matching records

        return None


    def printTree(tree):
        q = [(tree.root, 0)]
        while q:
            node, lvl = q.pop(0)
            print(f"Level {lvl} | leaf={node.is_leaf} | values={node.keys}")

            if not node.is_leaf:
                for child in node.keys:
                    q.append((child, lvl+1))
    
    def get_all_items(self):
        items = []

        # πήγαινε στο αριστερότερο leaf
        curr = self.root
        while not curr.is_leaf:
            curr = curr.keys[0]

    # διάσχιση leaf nodes
        while curr:
            for i, key_val in enumerate(curr.values):
                for record in curr.keys[i]:
                    items.append((key_val, record))
            curr = curr.Next

        return items
    
    def delete(self, sha_key):
        leaf = self.search(sha_key)

        for i, val in enumerate(leaf.values):
            if val == sha_key:
                leaf.values.pop(i)
                leaf.keys.pop(i)
                return
