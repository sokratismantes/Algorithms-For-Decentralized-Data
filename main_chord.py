from pathlib import Path
import argparse
import random
from threading import Thread, Lock

from data_read import load_and_preprocess
from chord import ChordRing, chord_hash


def read_titles_file(path: Path) -> list[str]:
    if not path.exists():
        return []
    titles = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            t = line.strip()
            if t:
                titles.append(t)
    return titles


def lookup_movie(title: str, ring: ChordRing, results: dict, lock: Lock):
    records, hops = ring.lookup(title)
    pop = records[0].get("popularity") if records else None
    with lock:
        results[title] = (pop, hops)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to dataset (.csv or .xlsx)")
    parser.add_argument("--k", type=int, default=3, help="Number of concurrent lookups")
    parser.add_argument("--seed", type=int, default=42, help="Seed used when sampling titles from dataset")
    parser.add_argument(
        "--titles",
        type=str,
        default="titles.txt",
        help="Optional titles file (one title per line). If present and non-empty, used instead of sampling.",
    )
    args = parser.parse_args()

    df = load_and_preprocess(Path(args.data))

    ring = ChordRing(m=40)

    space = 2 ** ring.m
    num_nodes = 10
    node_ids = [(i * space) // num_nodes for i in range(num_nodes)]
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

    # Prefer titles file if exists and non-empty
    titles_path = Path(args.titles)
    titles_to_lookup = read_titles_file(titles_path)

    if titles_to_lookup:
        source = f"titles file: {titles_path}"
    else:
        titles = df["title"].dropna().astype(str).tolist()
        if not titles:
            print("\nNo titles found in dataset.")
            return
        random.seed(args.seed)
        random.shuffle(titles)
        titles_to_lookup = titles
        source = f"dataset sampling (seed={args.seed})"

    titles_to_lookup = titles_to_lookup[: max(0, min(args.k, len(titles_to_lookup)))]
    print(f"\n[INFO] Using {len(titles_to_lookup)} titles from {source}")

    results = {}
    lock = Lock()
    threads = []
    for title in titles_to_lookup:
        t = Thread(target=lookup_movie, args=(title, ring, results, lock))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    print("\n=== Popularities of K movies ===")
    for title, (popularity, hops) in results.items():
        if popularity is not None:
            print(f'"{title}": popularity = {popularity}, hops = {hops}')
        else:
            print(f'"{title}" not found (hops = {hops})')


if __name__ == "__main__":
    main()
