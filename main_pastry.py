from pathlib import Path
import pandas as pd
import random
from threading import Thread
from collections import defaultdict
import statistics

from data_read import load_and_preprocess_csv
from pastry import PastryRing, pastry_hash

# for evaluation + plots (Chord vs Pastry) as in experiments
from experiments import run_for_N, plot_results


# ---------------- helpers ----------------
def _p95(values):
    if not values:
        return None
    s = sorted(values)
    idx = max(0, int(round(0.95 * (len(s) - 1))))
    return s[idx]

def _stats_line(name, values):
    if not values:
        return f"{name}: no data"
    return (f"{name}: count={len(values)} "
            f"avg={statistics.mean(values):.3f} "
            f"median={statistics.median(values):.3f} "
            f"p95={_p95(values):.3f} "
            f"min={min(values)} max={max(values)}")

def _pick_existing_title(df, preferred, fallback_idx=0):
    if preferred in set(df["title"].astype(str).tolist()):
        return preferred
    try:
        return str(df["title"].iloc[fallback_idx])
    except Exception:
        return preferred


# ---------------- load dataset (CSV) ----------------
project_root = Path(".").resolve()
file_path = project_root / "data_movies_clean.csv"
df = load_and_preprocess_csv(file_path, max_rows=50_000, seed=1)


# ---------------- demo: PASTRY ----------------
ring = PastryRing(m=40, leaf_size=8, btree_size=32)

max_hash = 2 ** 40 - 1
num_nodes = 10
# katanomh twn nodes
node_ids = [(i * max_hash) // num_nodes for i in range(1, num_nodes + 1)]

for nid in node_ids:
    ring.join_node(nid)

print("\n=== Initial Pastry Ring ===")
ring.print_nodes_summary()


# ----- Insert tis tainies -----
print("\nInserting movie records into Pastry...")
insert_hops = []
for _, row in df.iterrows():
    title = str(row["title"])
    key = pastry_hash(title, m=ring.m)
    ret = ring.insert(key, row.to_dict())
    hops = ret if isinstance(ret, int) else (ret[1] if isinstance(ret, tuple) and len(ret) >= 2 else 0)
    insert_hops.append(hops)

print("\n=== After inserting movies ===")
ring.print_nodes_summary()
print(_stats_line("Pastry INSERT hops", insert_hops))
# ----- Insert tis tainies -----


# ----- Eisagwgh neou node (dynamic JOIN) -----
new_node_id = 954584883608
join_ret = ring.join_node(new_node_id)
join_hops = join_ret[1] if isinstance(join_ret, tuple) and len(join_ret) >= 2 else 0
print("\n=== After joining new node (dynamic JOIN) ===")
ring.print_nodes_summary()
print(_stats_line("Pastry JOIN hops", [join_hops]))
# ----- Eisagwgh neou node -----


# ----- diagrafh enos node (dynamic LEAVE) -----
node_to_leave = node_ids[3]
leave_ret = ring.leave_node(node_to_leave)
leave_hops = leave_ret[1] if isinstance(leave_ret, tuple) and len(leave_ret) >= 2 else 0
print("\n=== After node leave (dynamic LEAVE) ===")
ring.print_nodes_summary()
print(_stats_line("Pastry LEAVE hops", [leave_hops]))
# ----- diagrafh enos node -----


# ----- update field tainias -----
title_to_update = _pick_existing_title(df, "History of a Crime")
field_to_update = "popularity"
new_value = 9.5

update_ret = ring.update_movie_field(title_to_update, field_to_update, new_value)
if isinstance(update_ret, tuple) and len(update_ret) >= 2 and isinstance(update_ret[1], int):
    update_hops = update_ret[1]
elif isinstance(update_ret, int):
    update_hops = update_ret
else:
    update_hops = 0

print(f'\n=== UPDATE field (title="{title_to_update}", field="{field_to_update}", new_value={new_value}) ===')
print(_stats_line("Pastry UPDATE hops", [update_hops]))
# ----- update field tainias -----


# ----- delete a movie title -----
title_to_delete = _pick_existing_title(df, "The Burning Mill", fallback_idx=1)
delete_ret = ring.delete_title(title_to_delete) if hasattr(ring, "delete_title") else None
if isinstance(delete_ret, int):
    delete_hops = delete_ret
elif isinstance(delete_ret, tuple) and len(delete_ret) >= 1 and isinstance(delete_ret[0], int):
    delete_hops = delete_ret[0]
else:
    # if your API uses delete(key) instead:
    try:
        key = pastry_hash(title_to_delete, m=ring.m)
        delete_ret2 = ring.delete(key)
        delete_hops = delete_ret2 if isinstance(delete_ret2, int) else (delete_ret2[1] if isinstance(delete_ret2, tuple) and len(delete_ret2) >= 2 else 0)
    except Exception:
        delete_hops = 0

print(f'\n=== DELETE title="{title_to_delete}" ===')
print(_stats_line("Pastry DELETE hops", [delete_hops]))
# ----- delete a movie title -----


# ----- dhmiourgia K threads gia K tainies -----
def lookup_movie(title, ring, results):  # euresh tou pediou pou theloume, default to popularity
    records, hops = ring.lookup(title)
    if records:
        results[title] = (records[0].get("popularity"), hops)
    else:
        results[title] = (None, hops)

# Lista K titlwn
titles_to_lookup = [
    _pick_existing_title(df, "Conquering the Skies", fallback_idx=2),
    _pick_existing_title(df, "Visit to Pompeii", fallback_idx=3),
    _pick_existing_title(df, "The Congress of Nations", fallback_idx=4),
]

# Apo8hkeush apotelesmatwn
results = {}
# Ekkinhsh twn nhmatwn
threads = []
for title in titles_to_lookup:
    t = Thread(target=lookup_movie, args=(title, ring, results))
    threads.append(t)
    t.start()
# Wait ola ta threads na teleiwsoun
for t in threads:
    t.join()

# Emfanish twn apotelesmatwn kai twn hops gia to kathe ena
print("\n=== Popularities of K movies (Pastry) ===")
lookup_hops = []
for title, (popularity, hops) in results.items():
    lookup_hops.append(hops)
    if popularity is not None:
        print(f'"{title}": popularity = {popularity}, hops = {hops}')
    else:
        print(f'"{title}" not found (hops = {hops})')
print(_stats_line("Pastry LOOKUP hops (K)", lookup_hops))
# ----- dhmiourgia K threads gia K tainies -----


# ---------------- Systematic evaluation + plots (Chord vs Pastry) ----------------
print("\n\n=== SYSTEMATIC EVALUATION + PLOTS (from experiments.py) ===")
outdir = project_root / "results_from_main_pastry"
outdir.mkdir(parents=True, exist_ok=True)

nodes_list = [16, 32, 64]
all_rows = []
for N in nodes_list:
    all_rows.extend(
        run_for_N(
            df=df,
            m=40,
            N=N,
            records_n=min(20_000, len(df)),
            updates_n=min(2_000, len(df)),
            queries_n=min(5_000, len(df)),
            deletes_n=min(2_000, len(df)),
            joins_n=10,
            leaves_n=10,
            seed=7,
        )
    )

res = pd.DataFrame(all_rows)
res.to_csv(outdir / "results.csv", index=True)
plot_results(res, outdir)

summary = (res.groupby(["operation", "protocol", "N"])["hops"]
           .agg(count="count",
                avg="mean",
                median="median",
                p95=lambda x: x.quantile(0.95),
                min="min",
                max="max")
           .reset_index()
           .sort_values(["operation", "N", "protocol"]))

summary.to_csv(outdir / "summary_stats.csv", index=False)

print("Saved:", outdir / "results.csv")
print("Saved:", outdir / "summary_stats.csv")
print("Plots saved:", ", ".join([str(outdir / f"{op}.png") for op in ["build","insert","update","lookup","delete","join","leave"]]))
