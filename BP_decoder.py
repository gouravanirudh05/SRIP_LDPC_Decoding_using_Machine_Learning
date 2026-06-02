import numpy as np

# ---------------------------------------------------
# Parity-check matrix for (7,4) Hamming code
# ---------------------------------------------------

H = np.array([
    [1,1,1,0,1,0,0],
    [1,1,0,1,0,1,0],
    [1,0,1,1,0,0,1]
])

m, n = H.shape

# ---------------------------------------------------
# Example transmitted codeword
# ---------------------------------------------------

tx_codeword = np.array([1,0,1,1,0,1,0])

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
# Channel LLRs
# ---------------------------------------------------

llr = 2 * rx_signal / (sigma**2)

# ---------------------------------------------------
# BP Initialization
# ---------------------------------------------------

iterations = 10

# Messages
msg_v_to_c = np.zeros((m, n))
msg_c_to_v = np.zeros((m, n))

# Initialize VN -> CN messages
for i in range(m):
    for j in range(n):
        if H[i,j] == 1:
            msg_v_to_c[i,j] = llr[j]

# ---------------------------------------------------
# BP Iterations
# ---------------------------------------------------

for _ in range(iterations):

    # ---------------------------------------------
    # Check node update
    # ---------------------------------------------

    for i in range(m):

        connected_vars = np.where(H[i] == 1)[0]

        for j in connected_vars:

            others = connected_vars[connected_vars != j]

            signs = np.prod(np.sign(msg_v_to_c[i, others]))

            minimum = np.min(np.abs(msg_v_to_c[i, others]))

            msg_c_to_v[i,j] = signs * minimum

    # ---------------------------------------------
    # Variable node update
    # ---------------------------------------------

    for j in range(n):

        connected_checks = np.where(H[:,j] == 1)[0]

        for i in connected_checks:

            others = connected_checks[connected_checks != i]

            msg_v_to_c[i,j] = (
                llr[j]
                + np.sum(msg_c_to_v[others, j])
            )

# ---------------------------------------------------
# Final LLRs
# ---------------------------------------------------

final_llr = np.copy(llr)

for j in range(n):

    connected_checks = np.where(H[:,j] == 1)[0]

    final_llr[j] += np.sum(msg_c_to_v[connected_checks, j])

# Hard decision
decoded_bits = (final_llr < 0).astype(int)

# ---------------------------------------------------
# Results
# ---------------------------------------------------

print("Transmitted codeword:")
print(tx_codeword)

print("\nReceived signal:")
print(rx_signal)

print("\nDecoded codeword:")
print(decoded_bits)