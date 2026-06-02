import numpy as np
import time
import matplotlib.pyplot as plt
from itertools import product

# ---------------------------------------------------
# Compare ML vs BP decoding complexity
# ---------------------------------------------------

k_values = range(2, 21)

ml_times = []
bp_times = []

iterations = 20

for k in k_values:

    # =================================================
    # ML DECODING
    # =================================================

    start_ml = time.time()

    # Exhaustive search over all codewords
    all_messages = list(product([0,1], repeat=k))

    # Dummy ML computation
    for msg in all_messages:
        _ = sum(msg)

    end_ml = time.time()

    ml_times.append(end_ml - start_ml)

    # =================================================
    # BP DECODING
    # =================================================

    # Assume code rate R = 1/2
    n = 2 * k

    variable_degree = 3

    edges = n * variable_degree

    messages_v_to_c = np.random.randn(edges)
    messages_c_to_v = np.random.randn(edges)

    start_bp = time.time()

    for _ in range(iterations):

        # Variable-to-check update
        messages_v_to_c = np.tanh(messages_c_to_v)

        # Check-to-variable update
        messages_c_to_v = np.tanh(messages_v_to_c)

    end_bp = time.time()

    bp_times.append(end_bp - start_bp)

    print(f"k={k} | ML={ml_times[-1]:.6f}s | BP={bp_times[-1]:.6f}s")

# ---------------------------------------------------
# Plot
# ---------------------------------------------------
plt.figure(figsize=(10,6))
plt.plot(k_values, ml_times, marker='o', label='ML Decoder')
plt.plot(k_values, bp_times, marker='s', label='BP Decoder')
plt.xlabel("Information Bits (k)")
plt.ylabel("Runtime (seconds)")
plt.title("ML vs BP Decoding Complexity")
plt.legend()
plt.grid(True)
plt.show()