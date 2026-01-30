from pathlib import Path
from threading import Thread, Lock
from data_read import load_and_preprocess
from chord import ChordRing

# Φόρτωσε δεδομένα
project_root = Path(__file__).resolve().parent.parent
file_path = project_root / "data" / "data_movies_clean_2.xlsx"
df = load_and_preprocess(file_path)

# Δημιουργία Chord Ring
ring = ChordRing(m=40, btree_size=32)

max_hash = 2 ** ring.m - 1
num_nodes = 10

# ομοιόμορφη κατανομή node IDs
node_ids = [(i * max_hash) // num_nodes for i in range(1, num_nodes + 1)]

# εισαγωγή nodes
for nid in node_ids:
    _, join_hops, moved = ring.join_node(nid)
    # προαιρετικό: print(f"join {nid}: hops={join_hops}, moved={moved}")

print("\n=== Initial Chord Ring ===")
ring.print_nodes_summary()

# ----- Insert ταινιών -----
print("\nInserting movie records into Chord...")
for _, row in df.iterrows():
    title = row["title"]
    ring.insert_title(title, row.to_dict())

print("\n=== After inserting movies ===")
ring.print_nodes_summary()

# ----- Concurrent K-lookups (popularities) -----
def lookup_movie(title, ring, results, lock: Lock):
    records, hops = ring.lookup(title)
    pop = records[0].get("popularity") if records else None
    with lock:
        results[title] = (pop, hops)

titles_to_lookup = [
    "Conquering the Skies",
    "Visit to Pompeii",
    "The Congress of Nations",
]

results = {}
lock = Lock()

threads = [Thread(target=lookup_movie, args=(t, ring, results, lock)) for t in titles_to_lookup]
for t in threads:
    t.start()
for t in threads:
    t.join()

print("\n=== Popularities of K movies (Chord) ===")
for title, (popularity, hops) in results.items():
    if popularity is not None:
        print(f'"{title}": popularity = {popularity}, hops = {hops}')
    else:
        print(f'"{title}" not found (hops = {hops})')
