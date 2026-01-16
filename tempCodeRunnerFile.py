from pathlib import Path
import argparse
import random

from data_read import load_and_preprocess
from chord import ChordRing, chord_hash


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to xlsx dataset")
    parser.add_argument("--m", type=int, default=40, help="Chord keyspace bits")
    parser.add_argument("--nodes", type=int, default=10, help="Number of nodes")
    args = parser.parse_args()

    df = load_and_preprocess(Path(args.data))

    ring = ChordRing(m=args.m)

    # uniform node ids 
    space = 2 ** ring.m
    node_ids = [(i * space) // args.nodes for i in range(args.nodes)]
    for nid in node_ids:
        ring.join_node(nid)

    print("\n=== Initial Chord Ring ===")
    ring.print_nodes_summary()

    print("\nInserting movie records into Chord...")
    for _, row in df.iterrows():
        title = str(row["title"])
        key = chord_hash(title, ring.m)
        ring.insert(key, row.to_dict())

    print("\n=== After inserting movies ===")
    ring.print_nodes_summary()

    # simple lookup test 
    titles = df["title"].dropna().astype(str).tolist()
    if titles:
        movie_title = random.choice(titles)
        records, hops = ring.lookup(movie_title)
        print(f'\nLookup: "{movie_title}" hops={hops}, found={bool(records)}')

    # optional join/leave sanity
    new_node_id = random.randrange(2 ** ring.m)
    _, join_hops = ring.join_node(new_node_id, measure_hops=True, verbose=True)
    print(f"\nJoin test: node={new_node_id}, total_redistribution_hops={join_hops}")

    leave_hops = ring.leave_node(new_node_id, measure_hops=True, verbose=True)
    print(f"\nLeave test: node={new_node_id}, total_redistribution_hops={leave_hops}")


if __name__ == "__main__":
    main()
