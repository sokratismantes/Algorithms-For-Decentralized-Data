# DHT Runner — Chord vs Pastry (με B+Tree αποθήκευση)

Το project συγκρίνει **Chord** και **Pastry** DHT πάνω σε dataset ταινιών (CSV), με αποθήκευση εγγραφών σε **B+Tree** ανά node. Περιλαμβάνει:
- CLI experiments για Chord: `main_chord.py`, `chord.py`
- CLI experiments για Pastry: `main_pastry.py`, `pastry.py` 
- GUI runner (Tkinter): `gui_dht.py`, `ui_helpers.py`, `experiments.py` 
- Loader / preprocessing του dataset: `data_read.py` 
- Plotters: `plot_chord.py`, `plot_pastry.py`
- B+ Tree struct: `b_tree.py`

---

### Python
- Python 3.x (προτείνεται 3.9+)

### Βιβλιοθήκες (pip)
- `pandas` (φόρτωση CSV)
- `matplotlib` (plots) 

Εγκατάσταση:
```bash
pip install pandas matplotlib

Εκτέλεση (CLI)
A) Chord experiment + plots

Κάνουμε run python main_chord.py - Θα φορτώσει το data_movies_clean.csv, θα κάνει joins/inserts/updates/deletes/lookups και στο τέλος θα ζητήσει K τίτλους για parallel lookup. 
Τα plots αποθηκεύονται σε φάκελο: results_from_main_chord/

B) Pastry experiment + plots

Κάνουμε run python main_pastry.py - Θα φορτώσει το data_movies_clean.csv, θα κάνει joins/inserts/updates/deletes/lookups και στο τέλος θα ζητήσει K τίτλους για parallel lookup. 
Τα plots αποθηκεύονται σε φάκελο: results_from_main_pastry/ 

4) Εκτέλεση (GUI)

Κάνουμε run python gui_dht.py - Στο παράθυρο βάζουμε Nodes (N), Lookups (K), Seed - Δίνουμε CSV path (default: ./data_movies_clean.csv) - (Προαιρετικά) γράφουμε τίτλους (έναν ανά γραμμή). Αν βάλουμε λιγότερους από K, τα υπόλοιπα επιλέγονται τυχαία - Πατάμε Run Chord + Pastry και βλέπουμε:
Summary metrics

Lookups (title → popularity, hops) 
