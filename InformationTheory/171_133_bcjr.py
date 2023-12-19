import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
import pandas as pd


def conv_encoder1711338(xx):  
    coded_bits = np.empty([0], order = 'C')
    # 两个寄存器u_1 u_2初始化为0
    uu1, uu2, uu3, uu4, uu5, uu6 = 0, 0, 0, 0, 0, 0
    for jj in xx:
        c_1 = (jj + uu1 + uu2 + uu3 + uu6) % 2
        c_2 = (jj + uu2 + uu3 + uu5 + uu6) % 2
        coded_bits = np.append(coded_bits, [c_1, c_2])
        # 更新寄存器
        uu6 = uu5
        uu5 = uu4
        uu4 = uu3
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
    # for xx, noise_real, noise_imag in zip(in_seq, real_part, imag_part):
    #     if xx == 0:
    #         out_seq.append([1, 0])
    #     elif xx == 1:
    #         out_seq.append([-1, 0])
    #     else:
    #         print('error at mod and transmit')
    #         pass
    #     pass
    return out_seq

class BCJR_4States():
    def __init__(self, in_seq):
        self.num_words = int( len(in_seq)/2 )
        self.demod_sym = np.asarray(in_seq)
        self.states = ['000000', '000001', '000010', '000011', '000100', '000101', '000110', '000111','001000', '001001','001010', '001011', '001100', '001101', '001110', '001111', '010000', '010001', '010010', '010011', '010100', '010101', '010110', '010111', '011000', '011001', '011010', '011011', '011100', '011101', '011110', '011111', '100000', '100001', '100010', '100011', '100100', '100101', '100110', '100111', '101000', '101001', '101010', '101011', '101100', '101101', '101110', '101111', '110000', '110001', '110010', '110011', '110100', '110101', '110110', '110111', '111000', '111001', '111010', '111011', '111100', '111101', '111110', '111111']
        self.mat_pc = list()
        self.mat_aa = np.zeros((self.num_words+1, self.states.__len__()))
        self.mat_aa[0][0] = 1
        self.mat_bb = np.zeros((self.num_words+1, self.states.__len__()))
        self.mat_bb[-1][0] = 1
        pass

    def state_transfer(self, input, now_state):
        u_1 = int(self.states[now_state][0])
        u_2 = int(self.states[now_state][1])
        u_3 = int(self.states[now_state][2])
        u_4 = int(self.states[now_state][3])
        u_5 = int(self.states[now_state][4])
        next_state = f'{input}{u_1}{u_2}{u_3}{u_4}{u_5}'
        return self.states.index(next_state)

    def encode_state(self, xx, state):
        codeword = np.empty([0], order='C')
        uu1 = int(self.states[state][0])
        uu2 = int(self.states[state][1])
        uu3 = int(self.states[state][2])
        uu4 = int(self.states[state][3])
        uu5 = int(self.states[state][4])
        uu6 = int(self.states[state][5])
        c_1 = (xx + uu1 + uu2 + uu3 + uu6) % 2
        c_2 = (xx + uu2 + uu3 + uu5 + uu6) % 2
        codeword = np.append(codeword, [c_1, c_2])
        return codeword

    def one_step_gamma(self, idx):  ## idx of codeword
        one_step_trellis = np.zeros((self.states.__len__(), self.states.__len__()))
        ## calculate the conditional probability P(u_idx = 0) P(y_idx^1 | c_idx^1) P(y_idx^2 | c_idx^2)
        ##index of curr_state
        for ii in range(self.states.__len__()):
            codeword0 = self.encode_state(0, ii).astype('int')
            next_state0 = self.state_transfer(0, ii)
            temp_prob = 0.5  ## assumed probability of transmitting 0
            for idx_digit, jj in enumerate(codeword0):  ## calculate 0.5 * P(y_idx^1 | c_idx^1) * P(y_idx^2 | c_idx^2)
                temp_prob *= self.mat_pc[jj][2*idx+idx_digit]
                pass
            one_step_trellis[ii][next_state0] += temp_prob

            codeword1 = self.encode_state(1, ii).astype('int')
            next_state1 = self.state_transfer(1, ii)
            temp_prob = 0.5
            for idx_digit, jj in enumerate(codeword1):
                temp_prob *= self.mat_pc[jj][2*idx+idx_digit]
                pass
            one_step_trellis[ii][next_state1] += temp_prob
            pass
        return one_step_trellis

    def cal_pc_aa_bb(self, noise_in_db):
        noise = 10**(noise_in_db/10)
        dist_to_zero = 1/(2*noise) * np.linalg.norm(self.demod_sym - [1,0], axis=1)**2
        self.mat_pc.append(1/np.sqrt(2*np.pi*noise) * np.exp(-dist_to_zero))  ## store p_c0
        dist_to_one = 1/(2*noise) * np.linalg.norm(self.demod_sym - [-1,0], axis=1)**2
        self.mat_pc.append(1/np.sqrt(2*np.pi*noise) * np.exp(-dist_to_one))   ## store p_c1

        for idx in range(1, self.num_words+1):  ## index of state instead of index of codeword
            ## idx is index of current state for mat_aa, and idx_revs is index of cu rrent state for mat_bb
            idx_revs = self.num_words - idx
            onestep_fowd_gamma = self.one_step_gamma(idx-1)
            onestep_back_gamma = self.one_step_gamma(idx_revs)

            self.mat_aa[idx] = np.dot(self.mat_aa[idx-1], onestep_fowd_gamma)
            self.mat_aa[idx] = self.mat_aa[idx]/sum( self.mat_aa[idx] )

            self.mat_bb[idx_revs] = np.dot(self.mat_bb[idx_revs+1], onestep_back_gamma.transpose())
            self.mat_bb[idx_revs] = self.mat_bb[idx_revs]/sum( self.mat_bb[idx_revs] )
            pass
        pass

    def decode(self, noise_in_db):
        self.cal_pc_aa_bb(noise_in_db)
        ## with A, B, and Gamma, we can now decode the codewords
        dec_bits = list()
        for idx in range(self.num_words):
            posteriori_prob = [list() for _ in range(2)]
            onestep_fowd_gamma = self.one_step_gamma(idx)
            for curr_state in range(self.states.__len__()):
                next_state0 = self.state_transfer(0, curr_state)
                posteriori_prob[0].append( self.mat_aa[idx][curr_state] * onestep_fowd_gamma[curr_state][next_state0] * self.mat_bb[idx+1][next_state0] )
                next_state1 = self.state_transfer(1, curr_state)
                posteriori_prob[1].append( self.mat_aa[idx][curr_state] * onestep_fowd_gamma[curr_state][next_state1] * self.mat_bb[idx+1][next_state1] )
                pass
            sum_posteriori = np.sum( np.asarray(posteriori_prob), axis=1 )
            dec_bits.append(np.argmax(sum_posteriori))
            pass
        return dec_bits


if __name__ == '__main__':
    np.random.seed(seed=926)
    num_registers = 6
    num_bits = 100000
    src_bits = np.random.randint(0, 2, num_bits)  ## generate source information bits
    src_bits = np.append(src_bits, np.zeros(num_registers))
    noise_sigma = np.arange(-5, -1, 0.5)  ## generate noise power in dB
    BER = list()
    statistics = [list() for _ in range(2)]

    for ii in trange(len(noise_sigma)):
        coded_bits = conv_encoder1711338(src_bits)  ## encode source bits
        recv_symbols = mod_and_trans(coded_bits, noise_sigma[ii])  ## modulate source bits to symbols and transmit over AWGN channel
        ## decode the demodulated bits
        bcjr_decoder = BCJR_4States(recv_symbols)
        dec_bits = bcjr_decoder.decode(noise_sigma[ii])
        BER.append( np.sum(src_bits != dec_bits) / (num_bits+2) )
        statistics[0].append( noise_sigma[ii] )
        statistics[1].append( BER[ii] )
        pass
    results = pd.DataFrame(np.asarray(statistics))
    FILE_PATH = './result/'
    board_dir = '171bcjr_dec'
    results.to_csv(FILE_PATH + board_dir + ".csv")

    plt.figure()
    plt.semilogy(noise_sigma, BER, 'o-', label='Conv Code $(171, 133)_8$ & bcjr Decoder')
    plt.xlabel('Noise Power (dB)')
    plt.ylabel('Bit Error Rate')
    plt.title('BER performance of Conv. Code over AWGN using BPSK')
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.show()
    plt.savefig('plot_(171,133)bcjr.png')