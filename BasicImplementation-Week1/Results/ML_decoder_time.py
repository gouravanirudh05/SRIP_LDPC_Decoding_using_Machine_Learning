import numpy as np
import time
import matplotlib.pyplot as plt
from itertools import product

# ---------------------------------------------
# Measure ML decoding time growth
# ---------------------------------------------

k_values = range(2, 21)   # information bits
times = []

for k in k_values:

    n = k + 3  # simple assumption

    # Start timer
    start_time = time.time()

    # Generate all possible messages
    all_messages = list(product([0,1], repeat=k))

    # Simulate ML exhaustive search
    # (dummy computation to imitate ML search)
    for msg in all_messages:
        _ = sum(msg)

    end_time = time.time()

    elapsed = end_time - start_time

    times.append(elapsed)

    print(f"k={k}, codewords={2**k}, time={elapsed:.6f} sec")

# ---------------------------------------------
# Plot results
# ---------------------------------------------

plt.figure(figsize=(8,5))

plt.plot(k_values, times, marker='o')

plt.xlabel("Number of Information Bits (k)")
plt.ylabel("Decoding Time (seconds)")
plt.title("Exponential Complexity of ML Decoding")

plt.grid(True)

plt.show()