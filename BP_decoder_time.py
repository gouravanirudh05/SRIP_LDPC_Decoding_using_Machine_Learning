import numpy as np
import time
import matplotlib.pyplot as plt

# ---------------------------------------------------
# Simulate BP decoding complexity scaling
# ---------------------------------------------------

n_values = range(100, 5001, 300)

times = []

iterations = 20

for n in n_values:

    # Assume sparse LDPC graph
    variable_degree = 3

    # Number of edges in sparse graph
    edges = n * variable_degree

    # Random messages
    messages_v_to_c = np.random.randn(edges)
    messages_c_to_v = np.random.randn(edges)

    start_time = time.time()

    # Simulated BP iterations
    for _ in range(iterations):

        # Variable-to-check updates
        messages_v_to_c = np.tanh(messages_c_to_v)

        # Check-to-variable updates
        messages_c_to_v = np.tanh(messages_v_to_c)

    end_time = time.time()

    elapsed = end_time - start_time

    times.append(elapsed)

    print(f"n={n}, edges={edges}, BP time={elapsed:.6f} sec")

# ---------------------------------------------------
# Plot
# ---------------------------------------------------

plt.figure(figsize=(8,5))

plt.plot(n_values, times, marker='o')

plt.xlabel("Code Length (n)")
plt.ylabel("Decoding Time (seconds)")
plt.title("Approximate Linear Complexity of BP Decoding")

plt.grid(True)

plt.show()