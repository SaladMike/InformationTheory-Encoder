import numpy as np
from tqdm import trange
import matplotlib

def hamming_dist(noisy_signal, enc_signal):
    dist = (noisy_signal[0] - enc_signal[0]) % 2 + (noisy_signal[1] - enc_signal[1]) % 2
    return dist

def henc_vertibi(input_bits, sigma):
    state = ['00', '01', '10', '11']

for ii in range (4):
    score_list[ii] = float('inf') * np.ones(tot_len + 1)
    pass
enc_bits = list()
u_2 = 0
u_1 = 0
init_state = f'{u_1}{u_2}'
init_state_idx = states.index(init_state)
for jj in input_bits:
    c1 = (jj + u_1 + u_2) % 2
    c2 = (jj + u_2) % 2
    enc_bits.append(c1)
    enc_bits.append(c2)
    u_2 = u_1
    u_1 = jj
    next_state = f'{u_1}{u_2}'
    next_state_idx = states.index(next_state)
    pass
return enc_bits

if __name__ = '__main__':
    num_bits = 100000
    bits = np.random.randint(0, 2, num_bits)
    snr_values = np.arange(0, 11, 1)
    ber_values = []
    for ii in trange(len(snr_values)):
        encoded_bits = henc_vertibi(bits, 10**snr_values[ii]/10)# 代码不全要补两个0