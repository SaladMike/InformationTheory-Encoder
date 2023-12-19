import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
import pandas as pd


def conv_encoder758(xx):  ## (7, 5)_8
    coded_bits = np.empty([0], order = 'C')
    # 两个寄存器u_1 u_2初始化为0
    uu1, uu2 = 0, 0
    for jj in xx:
        c_1 = (jj + uu1 + uu2) % 2
        c_2 = (jj + uu2) % 2
        coded_bits = np.append(coded_bits, [c_1, c_2])
        # 更新寄存器
        uu2 = uu1
        uu1 = jj
    return coded_bits

def mod_and_trans(in_seq, noise_in_db):
    out_seq = list()
    noise_pow = 10**(noise_in_db/10)
    real_part = np.sqrt(noise_pow) * np.random.randn(len(in_seq))
    imag_part = np.sqrt(noise_pow) * np.random.randn(len(in_seq))
    for xx, noise_real, noise_imag in zip(in_seq, real_part, imag_part):
        if xx == 0:
            out_seq.append([1+noise_real, 0+noise_imag])
        elif xx == 1:
            out_seq.append([-1+noise_real, 0+noise_imag])
        else:
            print('error at mod and transmit')
            pass
        pass
    return out_seq

def demod(in_seq):
    out_seq = list()
    distance0 = np.linalg.norm(np.array(in_seq) - np.array([1,0]), axis=1)
    distance1 = np.linalg.norm(np.array(in_seq) - np.array([-1,0]), axis=1)
    for dist0, dist1 in zip(distance0, distance1):
        if dist0 <= dist1:
            out_seq.append(0)
        elif dist0 > dist1:
            out_seq.append(1)
        else:
            print('error at demod')
            pass
        pass
    return out_seq


class Hard_Viterbi_4States():
    def __init__(self, in_seq):
        num_words = int( len(in_seq)/2 )
        self.demod_bits = in_seq
        self.accu_weight = np.array([np.inf * np.ones(num_words) for _ in range(4)])
        aa = np.inf * np.ones((4,1))
        aa[0] = 0
        self.accu_weight = np.concatenate((aa, self.accu_weight), axis=1 )
        self.states = ['00', '01', '10', '11']
        self.trace_block = []  ## first digit stores optimal previous state, second digit stores the input bit at the optimal state
        self.trace_back_list = []  ## store all track block

    def state_transfer(self, input, now_state):
        u_1 = int(self.states[now_state][0])
        next_state = f'{input}{u_1}'
        return self.states.index(next_state)

    def encode_state(self, xx, state):
        codeword = np.empty([0], order='C')
        uu1 = int(self.states[state][0])
        uu2 = int(self.states[state][1])
        c_1 = (xx + uu1 + uu2) % 2
        c_2 = (xx + uu2) % 2
        codeword = np.append(codeword, [c_1, c_2])
        return codeword

    def update_state(self, demod_word, curr_state, idx):
        # 输入的是 0
        codeword0 = self.encode_state(0, curr_state)
        next_state0 = self.state_transfer(0, curr_state)
        weight0 = sum( (demod_word-codeword0)%2 )
        # 模拟输入为0，且需要更新
        if self.accu_weight[curr_state][idx] + weight0 < self.accu_weight[next_state0][idx+1]:
            self.accu_weight[next_state0][idx+1] = self.accu_weight[curr_state][idx] + weight0
            self.trace_block[next_state0][0] = curr_state
            self.trace_block[next_state0][1] = 0
        # 输入的是 1
        codeword1 = self.encode_state(1, curr_state)
        next_state1 = self.state_transfer(1, curr_state)
        weight1 = sum( (demod_word-codeword1)%2 )
        # 模拟输入为1，且需要更新
        if self.accu_weight[curr_state][idx] + weight1 < self.accu_weight[next_state1][idx+1]:
            self.accu_weight[next_state1][idx+1] = self.accu_weight[curr_state][idx] + weight1
            self.trace_block[next_state1][0] = curr_state
            self.trace_block[next_state1][1] = 1
        pass

    def decode(self):
        demod_word = self.demod_bits[0:2]
        self.trace_block = [[-1, -1] for _ in range(4)]
        self.update_state(demod_word, curr_state=0, idx=0)
        self.trace_back_list.append(self.trace_block)
        # 开始之后的 y_block 更新
        for idx in range(2, int(len(self.demod_bits)), 2):
            demod_word = self.demod_bits[idx:idx+2]
            [self.update_state(demod_word, curr_state=state, idx=int(idx/2)) for state in range(len(self.states))]
            self.trace_back_list.append(self.trace_block)
            self.trace_block = [[-1, -1] for _ in range(4)]  # 在每次处理新的 demod_bits（即一对接收到的符号）时重置 trace_block
            pass
        # 完成前向，开始回溯
        state_trace_index = np.argmin(self.accu_weight[:, -1])
        xx = []
        for trace in range(len(self.trace_back_list)-1, -1, -1):
            xx.append(self.trace_back_list[trace][state_trace_index][1])
            state_trace_index = self.trace_back_list[trace][state_trace_index][0]
        xx = list(reversed(xx))
        return xx




if __name__ == '__main__':
    np.random.seed(seed=926)
    num_registers = 2
    num_bits = 100
    src_bits = np.random.randint(0, 2, num_bits)  ## generate source information bits
    src_bits = np.append(src_bits, np.zeros(num_registers))
    noise_sigma = np.arange(-5, -1, 0.5)  ## generate noise power in dB
    BER = list()
    statistics = [list() for _ in range(2)]

    for ii in trange(len(noise_sigma)):
        coded_bits = conv_encoder758(src_bits)  ## encode source bits
        recv_symbols = mod_and_trans(coded_bits, noise_sigma[ii])  ## modulate source bits to symbols and transmit over AWGN channel
        demod_bits = demod(recv_symbols)  ## demodulate the received symbols for hard Viterbi decode
        ## decode the demodulated bits
        hardViterbi_decoder = Hard_Viterbi_4States(demod_bits)
        dec_bits = hardViterbi_decoder.decode()
        BER.append( np.sum(src_bits != dec_bits) / (num_bits+2) )
        statistics[0].append( noise_sigma[ii] )
        statistics[1].append( BER[ii] )
        pass
    results = pd.DataFrame(np.asarray(statistics))
    FILE_PATH = './result/'
    board_dir = 'hard_viterbi'
    results.to_csv(FILE_PATH + board_dir + ".csv")

    # plt.figure()
    # plt.semilogy(noise_sigma, BER, 'o-', label='Conv Code $(7,5)_8$ & Hard Viterbi Decoder')
    # plt.xlabel('Noise Power (dB)')
    # plt.ylabel('Bit Error Rate')
    # plt.title('BER performance of Conv. Code over AWGN using BPSK')
    # plt.grid(True, which="both", ls="--")
    # plt.legend()
    # plt.show()


