import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
import pandas as pd


def conv_encoder15138(xx):  ## (15, 13)_8
    coded_bits = np.empty([0], order = 'C')
    # 两个寄存器u_1 u_2初始化为0
    uu1, uu2, uu3 = 0, 0, 0
    for jj in xx:
        c_1 = (jj + uu1 + uu3) % 2
        c_2 = (jj + uu2 + uu3) % 2
        coded_bits = np.append(coded_bits, [c_1, c_2])
        # 更新寄存器
        uu3 = uu2
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


class Soft_Viterbi_4States():
    def __init__(self, in_seq):
        num_words = int( len(in_seq)/2 )
        self.demod_bits = in_seq
        self.accu_weight = np.array([np.inf * np.ones(num_words) for _ in range(8)])
        aa = np.inf * np.ones((8,1))
        aa[0] = 0
        self.accu_weight = np.concatenate((aa, self.accu_weight), axis=1 )
        self.states = ['000', '001', '010', '011', '100', '101', '110', '111']
        self.trace_block = []  ## first digit stores optimal previous state, second digit stores the input bit at the optimal state
        self.trace_back_list = []  ## store all track block

    def state_transfer(self, input, now_state):
        u_1 = int(self.states[now_state][0])
        u_2 = int(self.states[now_state][1])
        next_state = f'{input}{u_1}{u_2}'
        return self.states.index(next_state)

    def encode_state(self, xx, state):
        codeword = np.empty([0], order='C')
        uu1 = int(self.states[state][0])
        uu2 = int(self.states[state][1])
        uu3 = int(self.states[state][2])
        c_1 = (xx + uu1 + uu3) % 2
        c_2 = (xx + uu2 + uu3) % 2
        codeword = np.append(codeword, [c_1, c_2])
        return codeword

    def cal_hamming_dist(self, demod_word, codeword):
        codesymbols = list()
        for ii in codeword:
            if int(ii) == 0:
                codesymbols.append([1,0])
            elif int(ii) == 1:
                codesymbols.append([-1, 0])
            else:
                print('error in calculating hamming')
            pass
        dist = np.linalg.norm( np.array(demod_word).flatten() - np.array(codesymbols).flatten() )
        return dist

    def update_state(self, demod_word, curr_state, idx):
        # 输入的是 0
        codeword0 = self.encode_state(0, curr_state)
        next_state0 = self.state_transfer(0, curr_state)
        weight0 = self.cal_hamming_dist(demod_word, codeword0)
        # 模拟输入为0，且需要更新
        if self.accu_weight[curr_state][idx] + weight0 < self.accu_weight[next_state0][idx+1]:
            self.accu_weight[next_state0][idx+1] = self.accu_weight[curr_state][idx] + weight0
            self.trace_block[next_state0][0] = curr_state
            self.trace_block[next_state0][1] = 0
        # 输入的是 1
        codeword1 = self.encode_state(1, curr_state)
        next_state1 = self.state_transfer(1, curr_state)
        weight1 = self.cal_hamming_dist(demod_word, codeword1)
        # 模拟输入为1，且需要更新
        if self.accu_weight[curr_state][idx] + weight1 < self.accu_weight[next_state1][idx+1]:
            self.accu_weight[next_state1][idx+1] = self.accu_weight[curr_state][idx] + weight1
            self.trace_block[next_state1][0] = curr_state
            self.trace_block[next_state1][1] = 1
        pass

    def decode(self):
        demod_word = self.demod_bits[0:2]
        self.trace_block = [[-1,-1] for _ in range(8)]
        self.update_state(demod_word, curr_state=0, idx=0)
        self.trace_back_list.append(self.trace_block)
        # 开始之后的 y_block 更新
        for idx in range(2, int(len(self.demod_bits)), 2):
            demod_word = self.demod_bits[idx:idx+2]
            [self.update_state(demod_word, curr_state=state, idx=int(idx/2)) for state in range(len(self.states))]
            self.trace_back_list.append(self.trace_block)
            self.trace_block = [[-1,-1] for _ in range(8)]  # 在每次处理新的 demod_bits（即一对接收到的符号）时重置 trace_block
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
    num_registers = 3
    num_bits = 100000
    src_bits = np.random.randint(0, 2, num_bits)  ## generate source information bits
    src_bits = np.append(src_bits, np.zeros(num_registers))
    noise_sigma = np.arange(-5, -1, 0.5)  ## generate noise power in dB
    BER = list()
    statistics = [list() for _ in range(2)]

    for ii in trange(len(noise_sigma)):
        coded_bits = conv_encoder15138(src_bits)  ## encode source bits
        recv_symbols = mod_and_trans(coded_bits, noise_sigma[ii])  ## modulate source bits to symbols and transmit over AWGN channel
        ## decode the demodulated bits
        hardViterbi_decoder = Soft_Viterbi_4States(recv_symbols)
        dec_bits = hardViterbi_decoder.decode()
        BER.append( np.sum(src_bits != dec_bits) / (num_bits+2) )
        statistics[0].append( noise_sigma[ii] )
        statistics[1].append( BER[ii] )
        pass
    results = pd.DataFrame(np.asarray(statistics))
    FILE_PATH = './result/'
    board_dir = '15soft_viterbi'
    results.to_csv(FILE_PATH + board_dir + ".csv")

    plt.figure()
    plt.semilogy(noise_sigma, BER, 'o-', label='Conv Code $(13,15)_8$ & Soft Viterbi Decoder')
    plt.xlabel('Noise Power (dB)')
    plt.ylabel('Bit Error Rate')
    plt.title('BER performance of Conv. Code over AWGN using BPSK')
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.show()
    plt.savefig('plot_(13, 15)s.png')


