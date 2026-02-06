# Topological Data Analysis Summary
- Observations processed: 547
- Assets included: 161
- Transfer entropy window (trading days): 40
- Transfer entropy sampling step (days): 1
- Transfer entropy snapshots ingested: 508
- TDA window size (snapshots): 25
- TDA window step (snapshots): 5
- TDA windows evaluated: 97
- Transfer entropy history length (k): 1
- Transfer entropy discretization bins: 12
- Symmetric transfer entropy (global): mean 1.3270, max 1.7959, min 0.4203
- Strongest average TE pairs:
  - 002716 CH Equity <-> 9988 HK Equity: 1.4581
  - 600862 CH Equity <-> 9988 HK Equity: 1.4455
  - 002179 CH Equity <-> 9988 HK Equity: 1.4448
  - 9988 HK Equity <-> 002085 CH Equity: 1.4396
  - 9988 HK Equity <-> 000423 CH Equity: 1.4384
- Rolling bottleneck distance (H1): mean 0.2670, max 0.6926, min 0.1522
- Betti numbers (H0): mean 160.00, max 160, min 160
- Betti numbers (H1): mean 108.32, max 150, min 64
- Mapper visualisation: exported to mapper_graph.html (144 nodes)

Betti numbers represent the count of topological features (components, loops, etc.) detected within each rolling window's persistent homology diagrams.