# Task 2: Heaps' Law

This task estimates Heaps' law (V = k * N^b) on the cleaned corpus using the same space-based tokenizer from Task 1. I streamed tokens in order, recorded vocabulary growth every 1,000 tokens, fit a log-log linear regression to obtain k and b, and saved both the fit and the sampled points. A log-log plot of observed points with the fitted curve was generated for visualization.
