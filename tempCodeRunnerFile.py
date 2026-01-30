"""
tempCodeRunnerFile.py

Προαιρετικό demo/δοκιμαστικό αρχείο (μπορείς να το αγνοήσεις στην παράδοση).
Για επίσημη αξιολόγηση/plots χρησιμοποίησε: experiments.py
"""

from pathlib import Path
from data_read import load_and_preprocess
from chord import ChordRing

project_root = Path(__file__).resolve().parent.parent
file_path = project_root / "data" / "data_movies_clean_2.xlsx"
df = load_and_preprocess(file_path)

ring = ChordRing(m=40, btree_size=32)

max_hash = 2 ** ring.m - 1
num_nodes = 10
node_ids = [(i * max_hash) // num_nodes for i in range(1, num_nodes + 1)]
for nid in node_ids:
    ring.join_node(nid)

for _, row in df.head(200).iterrows():
    ring.insert_title(row["title"], row.to_dict())

title = str(df.iloc[0]["title"])
records, hops = ring.lookup(title)
print(f"Lookup '{title}' -> hops={hops} found={bool(records)}")
