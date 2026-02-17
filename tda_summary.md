# Topological Data Analysis Summary
- Observations processed: 554
- Assets included: 161
- Point cloud method: umap
- TDA window size: 25
- TDA window step: 5
- TDA windows evaluated: 106
- Point cloud details:
  Components: 3
  Neighbors: 15
  Min dist: 0.1
  Metric: euclidean
  TDA window size: 25
  TDA step: 5
  Random state: 42
- Rolling bottleneck distance (H1): mean 0.1388, max 0.3450, min 0.0683
- Rolling diffusion distance (H1): mean 0.1713, max 0.5716, min 0.0163
- Betti numbers (H0): mean 160.00, max 160, min 160
- Betti numbers (H1): mean 50.60, max 70, min 38
- Betti numbers (H2): mean 7.10, max 15, min 0
- Euler characteristic: mean 116.50, max 125.00, min 98.00
- Persistent entropy:
  H0: mean 4.9982, max 5.0237, min 4.9735
  H1: mean 3.4900, max 3.8582, min 3.1534
  H2: mean 1.5053, max 2.3235, min 0.0000
- Mapper visualisation: exported to mapper_graph.html (43 nodes)

Betti numbers represent the count of topological features (components, loops, etc.) detected within each rolling window's persistent homology diagrams.