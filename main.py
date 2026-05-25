"""TODO(reproduction agent): experiment driver.

Should orchestrate the four approaches (Base-Hash, Base-Merge, Mat-View,
Merged-Idx) over TPC-H Q3, Q5, Q10 and the Invoice-extended variants Q3i,
Q5i, Q10i, on both LeanStore (B-tree) and RocksDB (LSM-tree) backends, and
emit the plots referenced by sections/experiments_revised.tex.
"""
