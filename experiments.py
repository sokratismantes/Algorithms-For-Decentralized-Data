import argparse
import random
from pathlib import Path
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
from time import perf_counter

from data_read import load_and_preprocess
from chord import ChordRing, chord_hash
from pastry import PastryRing, pastry_hash


def sample_titles(df, n):
    titles = df["title"].dropna().astype(str).tolist()
    if n >= len(titles):
        return titles
    return random.sample(titles, n)


def load_titles_file(path: str, k: int | None = None):
    titles = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            t = line.strip()
            if t:
                titles.append(t)
    if k is not None:
        titles = titles[:k]
    return titles


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def build_chord(df, n_nodes=16, m=20):
    ring = ChordRing(m=m)
    ids = random.sample(range(2**m), n_nodes)
    for nid in ids:
        ring.join_node(nid)

    total_hops = 0
    for _, row in df.iterrows():
        title = str(row["title"])
        key = chord_hash(title, ring.m)
        record = row.to_dict()
        total_hops += ring.insert(key, record)

    return ring, total_hops


def build_pastry(df, n_nodes=16, m=20):
    ring = PastryRing(m=m)
    ids = random.sample(range(2**m), n_nodes)
    for nid in ids:
        ring.join_node(nid)

    total_hops = 0
    for _, row in df.iterrows():
        title = str(row["title"])
        key = pastry_hash(title, m=ring.m)
        record = row.to_dict()
        total_hops += ring.insert(key, record)

    return ring, total_hops


def run_lookup(ring, titles, concurrent=False, workers=None):
    def one(t):
        _, h = ring.lookup(t)
        return h

    if not concurrent:
        return [one(t) for t in titles]

    if workers is None:
        workers = min(32, len(titles)) if titles else 1

    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(one, titles))


def run_insert(chord_ring, pastry_ring, n):
    chord_hops, pastry_hops = [], []
    for i in range(n):
        title = f"__synthetic_title_{i}__"
        rec = {"id": 10_000_000 + i, "title": title, "popularity": 1.0}

        ck = chord_hash(title, chord_ring.m)
        pk = pastry_hash(title, pastry_ring.m)

        chord_hops.append(chord_ring.insert(ck, rec))
        pastry_hops.append(pastry_ring.insert(pk, rec))

    return chord_hops, pastry_hops


def run_delete(chord_ring, pastry_ring, df, n):
    titles = df["title"].dropna().astype(str).tolist()
    random.shuffle(titles)
    titles = titles[: min(n, len(titles))]

    chord_hops, pastry_hops = [], []
    for title in titles:
        chord_hops.append(chord_ring.delete_title(title))     
        pastry_hops.append(pastry_ring.delete_title(title))   

    return chord_hops, pastry_hops
    

def run_update(chord_ring, pastry_ring, titles):
    chord_hops, pastry_hops = [], []
    for t in titles:
        chord_hops.append(chord_ring.update_movie_field(t, "popularity", 9999))
        pastry_hops.append(pastry_ring.update_movie_field(t, "popularity", 9999))
    return chord_hops, pastry_hops


def run_join_leave_split(chord_ring, pastry_ring, m):
    """
    Returns:
      (ch_join_overlay, ch_join_mig, pa_join_overlay, pa_join_mig,
       ch_leave_overlay, ch_leave_mig, pa_leave_overlay, pa_leave_mig)
    Requires chord.py & pastry.py to return split hops when measure_hops=True:
      join_node -> (node, (overlay_hops, migration_hops))
      leave_node -> (overlay_hops, migration_hops)
    """
    join_id = random.randrange(2**m)

    # join
    _, ch_join = chord_ring.join_node(join_id, measure_hops=True, verbose=False)
    _, pa_join = pastry_ring.join_node(join_id, measure_hops=True, verbose=False)

    ch_join_overlay, ch_join_mig = ch_join
    pa_join_overlay, pa_join_mig = pa_join

    # leave
    ch_leave = chord_ring.leave_node(join_id, measure_hops=True, verbose=False) or (0, 0)
    pa_leave = pastry_ring.leave_node(join_id, measure_hops=True, verbose=False) or (0, 0)

    ch_leave_overlay, ch_leave_mig = ch_leave
    pa_leave_overlay, pa_leave_mig = pa_leave

    return (
        ch_join_overlay, ch_join_mig, pa_join_overlay, pa_join_mig,
        ch_leave_overlay, ch_leave_mig, pa_leave_overlay, pa_leave_mig
    )


def plot_comparison(results):
    ops = list(results.keys())
    chord_vals = [results[o][0] for o in ops]
    pastry_vals = [results[o][1] for o in ops]

    x = range(len(ops))
    plt.figure()
    plt.bar([i - 0.2 for i in x], chord_vals, width=0.4, label="Chord")
    plt.bar([i + 0.2 for i in x], pastry_vals, width=0.4, label="Pastry")
    plt.xticks(list(x), ops, rotation=30, ha="right")
    plt.ylabel("hops")
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to dataset (.csv or .xlsx)")
    parser.add_argument("--nodes", type=int, default=16)
    parser.add_argument("--m", type=int, default=20)
    parser.add_argument("--q", type=int, default=200, help="random lookups")
    parser.add_argument("--nops", type=int, default=100, help="insert/delete operations sample size")

    parser.add_argument("--concurrent-lookup", action="store_true", help="Run lookups concurrently (threads)")
    parser.add_argument("--workers", type=int, default=None, help="Worker threads for concurrent lookup")

    parser.add_argument("--k", type=int, default=10, help="K for concurrent popularity detection")
    parser.add_argument("--titles-file", type=str, default=None, help="Path to file with K titles (one per line)")

    args = parser.parse_args()

    df = load_and_preprocess(Path(args.data))

    chord_ring, chord_build_hops = build_chord(df, n_nodes=args.nodes, m=args.m)
    pastry_ring, pastry_build_hops = build_pastry(df, n_nodes=args.nodes, m=args.m)

    titles_q = sample_titles(df, args.q)

    lookup_ch = run_lookup(chord_ring, titles_q, concurrent=args.concurrent_lookup, workers=args.workers)
    lookup_pa = run_lookup(pastry_ring, titles_q, concurrent=args.concurrent_lookup, workers=args.workers)

    ins_ch, ins_pa = run_insert(chord_ring, pastry_ring, args.nops)
    del_ch, del_pa = run_delete(chord_ring, pastry_ring, df, args.nops)

    upd_titles = sample_titles(df, min(50, args.q))
    upd_ch, upd_pa = run_update(chord_ring, pastry_ring, upd_titles)

    (
        ch_join_overlay, ch_join_mig, pa_join_overlay, pa_join_mig,
        ch_leave_overlay, ch_leave_mig, pa_leave_overlay, pa_leave_mig
    ) = run_join_leave_split(chord_ring, pastry_ring, args.m)

    # keep old metrics and add split join/leave
    results = {
        "build_total": (chord_build_hops, pastry_build_hops),
        "lookup_mean": (mean(lookup_ch), mean(lookup_pa)),
        "insert_mean": (mean(ins_ch), mean(ins_pa)),
        "delete_mean": (mean(del_ch), mean(del_pa)),
        "update_mean": (mean(upd_ch), mean(upd_pa)),

        "join_overlay": (ch_join_overlay, pa_join_overlay),
        "join_migration": (ch_join_mig, pa_join_mig),
        "leave_overlay": (ch_leave_overlay, pa_leave_overlay),
        "leave_migration": (ch_leave_mig, pa_leave_mig),
    }

    print("\n================ EXPERIMENT RESULTS ================")
    for op, (c, p) in results.items():
        print(f"{op:18s} | Chord={c:.3f} | Pastry={p:.3f}")
    print("====================================================\n")

    # K-movies concurrent popularity detection
    if args.titles_file:
        titles_k = load_titles_file(args.titles_file, k=args.k)
    else:
        titles_k = sample_titles(df, args.k)

    print(f"=== Concurrent popularity detection for K={len(titles_k)} movies ===")

    def chord_one(title):
        recs, hops = chord_ring.lookup(title)
        pop = recs[0].get("popularity") if recs else None
        print(f"[ChordRing] {title}: popularity={pop} (hops={hops})")
        return hops

    def pastry_one(title):
        recs, hops = pastry_ring.lookup(title)
        pop = recs[0].get("popularity") if recs else None
        print(f"[PastryRing] {title}: popularity={pop} (hops={hops})")
        return hops

    k_workers = args.workers
    if k_workers is None:
        k_workers = min(32, len(titles_k)) if titles_k else 1

    # concurrent chord
    t0 = perf_counter()
    with ThreadPoolExecutor(max_workers=k_workers) as ex:
        chord_hops_k = list(ex.map(chord_one, titles_k))
    dt = perf_counter() - t0
    print(
        f"[INFO] K-movies concurrent popularity detection finished in {dt:.3f}s "
        f"(K={len(titles_k)}, avg hops={mean(chord_hops_k):.3f})\n"
    )

    # concurrent pastry
    t0 = perf_counter()
    with ThreadPoolExecutor(max_workers=k_workers) as ex:
        pastry_hops_k = list(ex.map(pastry_one, titles_k))
    dt = perf_counter() - t0
    print(
        f"[INFO] K-movies concurrent popularity detection finished in {dt:.3f}s "
        f"(K={len(titles_k)}, avg hops={mean(pastry_hops_k):.3f})\n"
    )

    plot_comparison(results)


if __name__ == "__main__":
    main()
