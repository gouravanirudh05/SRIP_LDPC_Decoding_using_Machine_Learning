import numpy as np
from itertools import product

# ---------------------------------------------------
# Parameters
# ---------------------------------------------------

k = 4
n = 7

# Generator matrix for (7,4) Hamming code
G = np.array([
    [1,0,0,0,1,1,0],
    [0,1,0,0,1,0,1],
    [0,0,1,0,1,1,1],
    [0,0,0,1,0,1,1]
])

# ---------------------------------------------------
# Generate all possible codewords
# ---------------------------------------------------

all_messages = np.array(list(product([0,1], repeat=k)))

codebook = []

for msg in all_messages:
    codeword = np.mod(msg @ G, 2)
    codebook.append(codeword)

codebook = np.array(codebook)

# ---------------------------------------------------
# Example transmitted message
# ---------------------------------------------------

tx_bits = np.array([1,0,1,1])

tx_codeword = np.mod(tx_bits @ G, 2)

# BPSK modulation
tx_signal = 1 - 2 * tx_codeword

# ---------------------------------------------------
# AWGN channel
# ---------------------------------------------------

snr_db = 4

snr_linear = 10**(snr_db/10)

sigma = np.sqrt(1/(2*snr_linear))

noise = sigma * np.random.randn(n)

rx_signal = tx_signal + noise

# ---------------------------------------------------
# ML Decoding
# ---------------------------------------------------

# Convert all codewords to BPSK
bpsk_codebook = 1 - 2 * codebook

# Euclidean distance
distances = np.sum((bpsk_codebook - rx_signal)**2, axis=1)

# Find closest codeword
best_index = np.argmin(distances)

decoded_codeword = codebook[best_index]

# ---------------------------------------------------
# Results
# ---------------------------------------------------

print("Transmitted codeword:")
print(tx_codeword)

print("\nReceived signal:")
print(rx_signal)

print("\nDecoded codeword:")
print(decoded_codeword)