import hashlib
import random
from typing import List, Optional, Tuple, Union

from b_tree import BPlusTree


def _norm_title(x) -> str:
    return str(x).strip()


def chord_hash(value, m: int = 40) -> int:
    """SHA-1 hash -> int mod 2^m."""
    v = _norm_title(value)  # normalize
    h = hashlib.sha1(str(v).encode()).hexdigest()
    return int(h, 16) % (2**m)


class ChordNode:
    def __init__(self, node_id: int, m: int, btree_size: int = 32):
        self.node_id = node_id
        self.m = m
        self.btree = BPlusTree(btree_size)

        self.successor: Optional["ChordNode"] = None
        self.predecessor: Optional["ChordNode"] = None

        self.finger: List[Optional["ChordNode"]] = [None] * m

    def __repr__(self) -> str:
        return f"Node({self.node_id})"


class ChordRing:
    def __init__(self, m: int = 40, btree_size: int = 32):
        self.m = m
        self.btree_size = btree_size
        self.nodes: List[ChordNode] = []

    # helpers 
    def _random_start(self) -> Optional[ChordNode]:
        """Random start node for consistent hop measurement policy."""
        return random.choice(self.nodes) if self.nodes else None

    def _in_interval(self, key: int, start: int, end: int) -> bool:
        """Check if key in (start, end] on circular ring modulo 2^m."""
        if start < end:
            return start < key <= end
        if start > end:
            return key > start or key <= end
        return True  # start to end full circle

    def _update_links(self) -> None:
        n = len(self.nodes)
        for i, node in enumerate(self.nodes):
            node.successor = self.nodes[(i + 1) % n]
            node.predecessor = self.nodes[(i - 1) % n]

    # routing 
    def closest_preceding_finger(self, node: ChordNode, key: int) -> ChordNode:
        for i in reversed(range(self.m)):
            f = node.finger[i]
            if f and self._in_interval(f.node_id, node.node_id, key):
                return f
        return node

    def find_successor_linear(self, key: int) -> ChordNode:
        """Safe linear successor search (nodes are sorted by id)."""
        for node in self.nodes:
            if key <= node.node_id:
                return node
        return self.nodes[0]

    def find_successor(self, key: int, start_node: Optional[ChordNode] = None) -> Tuple[ChordNode, int]:
        """Find successor of key using finger tables. Returns (successor, hops)."""
        if not self.nodes:
            raise RuntimeError("No nodes in Chord ring")

        if start_node is None:
            start_node = self.nodes[0]

        hops = 0
        curr = start_node
        visited = set()
        max_hops = max(8, len(self.nodes) * 4)

        if curr.successor is None:
            self._update_links()

        while not self._in_interval(key, curr.node_id, curr.successor.node_id):
            if curr in visited or hops > max_hops:
                return self.find_successor_linear(key), hops

            visited.add(curr)
            nxt = self.closest_preceding_finger(curr, key)
            if nxt is curr:
                nxt = curr.successor

            curr = nxt
            hops += 1

        return curr.successor, hops

    # finger tables 
    def init_finger_table(self, node: ChordNode) -> None:
        max_id = 2**self.m
        for i in range(self.m):
            start = (node.node_id + 2**i) % max_id
            node.finger[i] = self.find_successor_linear(start)

    def fix_all_fingers(self) -> None:
        for node in self.nodes:
            self.init_finger_table(node)

    # data ops
    def insert(self, key: int, record: dict) -> int:
        start = self._random_start()
        node, hops = self.find_successor(key, start_node=start)
        node.btree.insert(record, key)
        return hops

    def lookup(self, movie_title: str) -> Tuple[Optional[List[dict]], int]:
        title_n = _norm_title(movie_title)  
        key = chord_hash(title_n, self.m)
        start = self._random_start()
        node, hops = self.find_successor(key, start_node=start)

        recs = node.btree.search_key(key)  
        recs = [r for r in recs if _norm_title(r.get("title", "")) == title_n]  
        return (recs if recs else None), hops  

    def delete_key(self, key: int) -> int:
        start = self._random_start()
        node, hops = self.find_successor(key, start_node=start)
        node.btree.delete(key)
        return hops

    def delete_title(self, title: str) -> int:
        title_n = _norm_title(title)  
        key = chord_hash(title_n, self.m)
        start = self._random_start()
        node, hops = self.find_successor(key, start_node=start)
        node.btree.delete_title(key, title_n)  
        return hops

    def update_movie_field(self, title: str, field: str, new_value) -> int:
        title_n = _norm_title(title) 
        key = chord_hash(title_n, self.m)
        start = self._random_start()
        node, hops = self.find_successor(key, start_node=start)

        records = node.btree.search_key(key)
        records = [r for r in records if _norm_title(r.get("title", "")) == title_n]  
        if not records:
            print(f'Movie "{title}" not found.')
            return 0

        record = records[0]
        old_value = record.get(field, None)
        record[field] = new_value
        print(f'Updated "{title}": {field} changed from {old_value} to {new_value} (hops={hops})')
        return hops

    # membership 
    def join_node(
        self,
        node_id: int,
        measure_hops: bool = False,
        verbose: bool = True,
    ) -> Union[ChordNode, Tuple[ChordNode, Tuple[int, int]]]:
        """
        Adds a node to the ring.

        If measure_hops=True, returns:
          (new_node, (overlay_hops, migration_hops))

        In this simulation model:
          overlay_hops = 0 (join is centralized here),
          migration_hops = total hops used while migrating keys to the new node.
        """
        new_node = ChordNode(node_id, self.m, btree_size=self.btree_size)
        self.nodes.append(new_node)
        self.nodes.sort(key=lambda n: n.node_id)
        self._update_links()

        # Stabilize routing before measuring migration hops
        self.fix_all_fingers()

        migration_hops = 0
        if len(self.nodes) > 1:
            migration_hops = self._redistribute_keys(new_node, verbose=verbose)

        if measure_hops:
            return new_node, (0, migration_hops)
        return new_node

    def _redistribute_keys(self, new_node: ChordNode, verbose: bool = True) -> int:
        """Move keys from successor to new_node if they belong to its interval."""
        pred = new_node.predecessor
        succ = new_node.successor

        items = succ.btree.get_all_items()
        total_hops = 0
        moved = 0

        for sha_key, record in items:
            if self._in_interval(sha_key, pred.node_id, new_node.node_id):
                # start from the node that currently holds the key 
                _, hops = self.find_successor(sha_key, start_node=succ)
                total_hops += hops
                moved += 1

                new_node.btree.insert(record, sha_key)
                succ.btree.delete(sha_key)

        if verbose:
            print(f"\nNode join: moved {moved} keys, total migration hops = {total_hops}")

        return total_hops

    def leave_node(
        self,
        node_id: int,
        measure_hops: bool = False,
        verbose: bool = True,
    ) -> Union[None, Tuple[int, int]]:
        """
        Removes a node and re-places its keys according to post-leave topology.

        If measure_hops=True, returns (overlay_hops, migration_hops).
        overlay_hops = 0 (centralized membership update in this simulation).
        migration_hops = total hops used to route keys to their new owners.
        """
        node = next((n for n in self.nodes if n.node_id == node_id), None)
        if not node:
            if verbose:
                print(f"Node {node_id} not found")
            return (0, 0) if measure_hops else None

        if len(self.nodes) == 1:
            self.nodes.remove(node)
            return (0, 0) if measure_hops else None

        leaving_items = list(node.btree.get_all_items())

        # capture a valid start node for migration before removing node from the ring
        handoff_start = node.successor

        # Remove node, stabilize routing
        self.nodes.remove(node)
        self._update_links()
        self.fix_all_fingers()

        migration_hops = 0
        moved = 0
        for sha_key, record in leaving_items:
            # start from successor of leaving node 
            owner, hops = self.find_successor(sha_key, start_node=handoff_start)
            migration_hops += hops
            moved += 1
            owner.btree.insert(record, sha_key)

        if verbose:
            print(f"\nNode leave: moved {moved} keys, total migration hops = {migration_hops}")

        return (0, migration_hops) if measure_hops else None

    # reporting 
    def print_nodes_summary(self) -> None:
        for n in self.nodes:
            items = n.btree.get_all_items()
            print(f"{n} keys = {len(items)}")
