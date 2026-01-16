from __future__ import annotations

import hashlib
import random
from typing import Iterable, List, Optional, Tuple, Union

from b_tree import BPlusTree


def _norm_title(x) -> str:
    return str(x).strip()


def pastry_hash(value, m: int = 40) -> int:
    """SHA-1 hash -> int mod 2^m."""
    v = _norm_title(value)  
    h = hashlib.sha1(str(v).encode()).hexdigest()
    return int(h, 16) % (2**m)


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


class PastryNode:
    """
    Canonical (digit-based) Pastry node:
      - IDs/keys are represented in base 2^b (digits)
      - routing table: rows = ceil(m/b), cols = 2^b
      - leaf set: nearest neighbors (simulation-friendly)
    """

    def __init__(
        self,
        node_id: int,
        m: int,
        b: int = 4,
        btree_size: int = 32,
        leaf_size: int = 8,
    ):
        self.id = node_id
        self.m = m
        self.b = b
        self.base = 2**b
        self.space = 2**m

        self.btree = BPlusTree(btree_size)
        self.leaf_size = leaf_size

        self.digit_len = _ceil_div(m, b)
        self.id_digits = self._to_digits(self.id)

        self.route_table: List[List[Optional["PastryNode"]]] = [
            [None for _ in range(self.base)] for _ in range(self.digit_len)
        ]
        self.leaf_set: List["PastryNode"] = []

    def _to_digits(self, x: int) -> Tuple[int, ...]:
        digits = [0] * self.digit_len
        for i in range(self.digit_len - 1, -1, -1):
            digits[i] = x % self.base
            x //= self.base
        return tuple(digits)

    def __repr__(self) -> str:
        return f"PNode({self.id})"

    def distance_to(self, key: int) -> int:
        diff = abs(self.id - key)
        return min(diff, self.space - diff)


class PastryRing:
    """
    Canonical digit-based Pastry overlay with a simulation-friendly membership model.

    Important note for experiments:
    - join/leave rebuild routing tables globally (centralized simulation),
      so overlay_hops = 0 in this model.
    - migration_hops counts routing hops incurred while moving keys
      so that each key ends up at the node reached by Pastry routing.
    """

    def __init__(self, m: int = 40, b: int = 4, leaf_size: int = 8, btree_size: int = 32):
        self.m = m
        self.b = b
        self.base = 2**b
        self.digit_len = _ceil_div(m, b)
        self.space = 2**m

        self.leaf_size = leaf_size
        self.btree_size = btree_size
        self.nodes: List[PastryNode] = []

    # helpers 
    def _id_to_digits(self, x: int) -> Tuple[int, ...]:
        digits = [0] * self.digit_len
        for i in range(self.digit_len - 1, -1, -1):
            digits[i] = x % self.base
            x //= self.base
        return tuple(digits)

    @staticmethod
    def _prefix_len(a: Tuple[int, ...], b: Tuple[int, ...]) -> int:
        i = 0
        for da, db in zip(a, b):
            if da != db:
                break
            i += 1
        return i

    def _iter_all_items(self) -> Iterable[Tuple[PastryNode, int, dict]]:
        for n in self.nodes:
            for k, rec in n.btree.get_all_items():
                yield n, k, rec

    # topology rebuild 
    def _rebuild_all(self) -> None:
        if not self.nodes:
            return

        self.nodes.sort(key=lambda x: x.id)
        n_nodes = len(self.nodes)

        for node in self.nodes:
            node.m = self.m
            node.b = self.b
            node.base = self.base
            node.space = self.space
            node.leaf_size = self.leaf_size
            node.digit_len = self.digit_len
            node.id_digits = node._to_digits(node.id)
            node.route_table = [[None for _ in range(self.base)] for _ in range(self.digit_len)]

        # nearest neighbors in circular order
        for idx, node in enumerate(self.nodes):
            neighbors: List[PastryNode] = []
            step = 1
            target = min(self.leaf_size, n_nodes - 1)
            while len(neighbors) < target and step < n_nodes:
                left = self.nodes[(idx - step) % n_nodes]
                right = self.nodes[(idx + step) % n_nodes]
                if left is not node and left not in neighbors:
                    neighbors.append(left)
                if len(neighbors) < target and right is not node and right not in neighbors:
                    neighbors.append(right)
                step += 1
            node.leaf_set = neighbors[:target]

        # routing tables 
        for node in self.nodes:
            for other in self.nodes:
                if other is node:
                    continue
                pref = self._prefix_len(node.id_digits, other.id_digits)
                if pref >= self.digit_len:
                    continue
                next_digit = other.id_digits[pref]
                existing = node.route_table[pref][next_digit]
                if existing is None:
                    node.route_table[pref][next_digit] = other
                else:
                    if other.distance_to(node.id) < existing.distance_to(node.id):
                        node.route_table[pref][next_digit] = other

    # routing
    def _route(self, start_node: PastryNode, key_int: int, key_digits: Tuple[int, ...]) -> Tuple[PastryNode, int]:
        hops = 0
        curr = start_node
        visited = set()
        max_steps = max(8, len(self.nodes) * 4)

        for _ in range(max_steps):
            if curr in visited:
                return curr, hops
            visited.add(curr)

            # leaf-set numeric range check 
            leaf_candidates = [curr] + list(curr.leaf_set)
            ids = [n.id for n in leaf_candidates]
            lo, hi = min(ids), max(ids)

            if lo <= key_int <= hi:
                best = min(leaf_candidates, key=lambda n: abs(n.id - key_int))
                if best is curr:
                    return curr, hops
                curr = best
                hops += 1
                continue

            pref = self._prefix_len(curr.id_digits, key_digits)
            if pref < self.digit_len:
                next_digit = key_digits[pref]
                rt = curr.route_table[pref][next_digit]
                if rt is not None and rt is not curr:
                    curr = rt
                    hops += 1
                    continue

            # fallback among candidates
            candidates = set(curr.leaf_set)
            for row in curr.route_table:
                for n in row:
                    if n is not None:
                        candidates.add(n)

            best = curr
            best_pref = pref
            best_dist = abs(curr.id - key_int)

            for c in candidates:
                p = self._prefix_len(c.id_digits, key_digits)
                d = abs(c.id - key_int)
                if p > best_pref and d <= best_dist:
                    best = c
                    best_pref = p
                    best_dist = d
                elif p == best_pref and d < best_dist:
                    best = c
                    best_dist = d

            if best is curr:
                return curr, hops

            curr = best
            hops += 1

        return curr, hops

    def _route_owner(self, key_int: int, start_node: Optional[PastryNode] = None) -> Tuple[PastryNode, int]:
        if not self.nodes:
            raise RuntimeError("No nodes in pastry ring")
        if start_node is None:
            start_node = random.choice(self.nodes)
        key_digits = self._id_to_digits(key_int)
        return self._route(start_node, key_int, key_digits)

    # migration / rebalance 
    def _rebalance_all_keys(self, measure_hops: bool, verbose: bool) -> int:
        """
        Ensure each key is stored at the node reached by Pastry routing.
        Returns migration hops (sum of routing hops during moves).
        """
        if not self.nodes:
            return 0

        total_hops = 0
        moved = 0

        all_items: List[Tuple[PastryNode, int, dict]] = list(self._iter_all_items())

        for src_node, k, rec in all_items:
            current_records = src_node.btree.search_key(k)
            if not current_records:
                continue

            key_digits = self._id_to_digits(k)
            dst_node, hops = self._route(src_node, k, key_digits)

            if dst_node is not src_node:
                dst_node.btree.insert(rec, k)
                src_node.btree.delete(k)
                moved += 1
                if measure_hops:
                    total_hops += hops

        if verbose:
            print(f"Pastry rebalance: moved {moved} keys (total_hops={total_hops})")

        return total_hops

    # membership 
    def join_node(
        self,
        node_id: int,
        measure_hops: bool = False,
        verbose: bool = True,
    ) -> Union[PastryNode, Tuple[PastryNode, Tuple[int, int]]]:
        """
        Adds a node.

        If measure_hops=True, returns:
          (new_node, (overlay_hops, migration_hops))

        In this simulation model:
          overlay_hops = 0 (global rebuild is centralized),
          migration_hops = hops used while rebalancing keys after join.
        """
        new_node = PastryNode(
            node_id,
            self.m,
            b=self.b,
            btree_size=self.btree_size,
            leaf_size=self.leaf_size,
        )
        self.nodes.append(new_node)
        self._rebuild_all()

        if len(self.nodes) == 1:
            if measure_hops:
                return new_node, (0, 0)
            return new_node

        migration_hops = self._rebalance_all_keys(measure_hops=measure_hops, verbose=verbose)

        if measure_hops:
            return new_node, (0, migration_hops)
        return new_node

    def leave_node(
        self,
        node_id: int,
        measure_hops: bool = False,
        verbose: bool = True,
    ) -> Union[None, Tuple[int, int]]:
        """
        Removes a node.

        If measure_hops=True, returns (overlay_hops, migration_hops).
        overlay_hops = 0 in this centralized rebuild model.
        migration_hops includes routing hops while moving keys from the leaving node
        plus rebalance hops.
        """
        node = next((n for n in self.nodes if n.id == node_id), None)
        if node is None:
            if verbose:
                print(f"Node {node_id} not found")
            return (0, 0) if measure_hops else None

        leaving_items = list(node.btree.get_all_items())

        self.nodes.remove(node)
        self._rebuild_all()

        migration_hops = 0
        moved = 0

        if self.nodes:
            # Move keys from leaving node to routed owner
            for k, rec in leaving_items:
                handoff = random.choice(self.nodes)
                owner, hops = self._route_owner(k, start_node=handoff)
                owner.btree.insert(rec, k)
                moved += 1
                if measure_hops:
                    migration_hops += hops

            # Full consistency after leave
            migration_hops += self._rebalance_all_keys(measure_hops=measure_hops, verbose=False)

        if verbose:
            print(f"Node leave: moved {moved} keys from leaving node (total_hops={migration_hops})")

        return (0, migration_hops) if measure_hops else None

    # data ops 
    def insert(self, key_int: int, record: dict, start_node: Optional[PastryNode] = None) -> int:
        owner, hops = self._route_owner(key_int, start_node=start_node)
        owner.btree.insert(record, key_int)
        return hops

    def lookup(self, movie_title: str, start_node: Optional[PastryNode] = None) -> Tuple[Optional[List[dict]], int]:
        if not self.nodes:
            return None, 0

        title_n = _norm_title(movie_title)  
        key = pastry_hash(title_n, m=self.m)
        owner, hops = self._route_owner(key, start_node=start_node)

        recs = owner.btree.search_key(key)
        recs = [r for r in recs if _norm_title(r.get("title", "")) == title_n]  
        return (recs if recs else None), hops

    def delete_key(self, key_int: int, start_node: Optional[PastryNode] = None) -> int:
        owner, hops = self._route_owner(key_int, start_node=start_node)
        owner.btree.delete(key_int)
        return hops

    def delete_title(self, title: str, start_node: Optional[PastryNode] = None) -> int:
        title_n = _norm_title(title)  
        key = pastry_hash(title_n, m=self.m)
        owner, hops = self._route_owner(key, start_node=start_node)
        owner.btree.delete_title(key, title_n)  
        return hops

    def update_movie_field(self, title: str, field: str, new_value) -> int:
        if not self.nodes:
            raise RuntimeError("No nodes in pastry ring")

        title_n = _norm_title(title)  
        key = pastry_hash(title_n, m=self.m)
        start_node = random.choice(self.nodes)
        owner, hops = self._route_owner(key, start_node=start_node)

        records = owner.btree.search_key(key)
        records = [r for r in records if _norm_title(r.get("title", "")) == title_n]  
        if not records:
            print(f'Movie "{title}" not found')
            return 0

        rec = records[0]
        old = rec.get(field)
        rec[field] = new_value
        print(f'Updated "{title}" field "{field}": {old} -> {new_value} (hops={hops})')
        return hops

    # reporting 
    def print_nodes_summary(self) -> None:
        for n in self.nodes:
            items = n.btree.get_all_items()
            print(f"{n} keys = {len(items)}")
