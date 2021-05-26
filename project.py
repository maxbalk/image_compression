import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

from copy import deepcopy

# define filter coefficients
lo_forward = np.array([-1/8, 2/8, 6/8, 2/8, -1/8])
hi_forward = np.array([-1/2, 1, -1/2])
hi_inverse = np.array([1/2, 1, 1/2])
lo_inverse = np.array([-1/8, -2/8, 6/8, -2/8, -1/8])

def transform(x, filter):
    extension = math.floor(len(filter)/2)
    x = np.insert(x, 0, x[1:1+extension][::-1])
    x = np.append(x, x[len(x)-(extension)-1:-1][::-1])
    out = []
    for i in range(extension, len(x)-extension):
        out.append(np.dot(filter, x[i-extension:i+extension+1]))
    return np.array(out)

def decompose(img):
    rows = []
    for row in img:
        # apply filter
        h0 = transform(row, lo_forward)
        h1 = transform(row, hi_forward)
        # downsample 
        h0 = h0[0::2]
        h1 = h1[0::2]
        rows.append(np.append(h0, h1))
    return np.array(rows)

def analysis(img):
    bands = decompose(img)
    bands = np.rot90(bands, 1)
    bands = decompose(bands)
    bands = np.rot90(bands, -1)
    return bands


def reconstruct(img):
    for i, row in enumerate(img):
        g0 = row[:int(len(row)/2)]
        g1 = row[int(len(row)/2):]
        # upsample. do first and second halves serparately 
        g0 = np.insert(g0, range(1,len(g0)+1), 0)
        g1 = np.insert(g1, range(1,len(g1)+1), 0)
        # apply filter
        g0 = transform(g0, hi_inverse)
        g1 = transform(g1, lo_inverse)
        # add row values together, not append
        img[i] = np.sum([g0,g1], axis=0)
    return img

def synthesis(img):
    bands = np.rot90(img, 1)
    bands = reconstruct(bands)
    bands = np.rot90(bands, -1)
    bands = reconstruct(bands)
    return bands

def forwardDWT(y_out):
    for i in range(decomp_level):    
        y_out[:int(x_dim/(2**i)),:int(y_dim/(2**i))] = analysis(y_out[:int(x_dim/(2**i)),:int(y_dim/(2**i))])
    return y_out

def inverseDWT(x_hat):
    for i in reversed(range(decomp_level)):
        x_hat[:int(x_dim/(2**i)),:int(y_dim/(2**i))] = synthesis(x_hat[:int(x_dim/(2**i)),:int(y_dim/(2**i))])
    return x_hat

amp_table = {
    1: 1,
    2: 3,
    3: 7,
    4: 15,
    5: 31,
    6: 63,
    7: 127,
    8: 255,
    9: 511,
    10: 1023
}

def amplitude(char):
    group = 1
    for key, val in amp_table.items():
        if abs(char) > val:
            group += 1
        else:
            if char < 0:
               return (group, val - abs(char))
            return (group, char)

def run_size(img):
    for row in img:
        n_zero = 0
        for char in row:
            if n_zero >= max_run:
                symbols.append( (max_run,0) )
                n_zero = 0
            elif char != 0:
                group, amp = amplitude(char)
                symbols.append((n_zero, group))
                bin = f"{int(amp):b}"
                for i in range(group - len(f"{int(amp):b}")):
                    bin = '0' + bin
                amps.append(bin)
                n_zero = 0
            else:
                n_zero += 1
        if n_zero > 0:
            symbols.append( (0,0) )

def symbol_dist(symbols):
    symbol_distribution = {}
    for symbol in symbols:
        try:
            symbol_distribution[symbol] +=1
        except KeyError:
            symbol_distribution[symbol] = 1

    for key, val in symbol_distribution.items():
        symbol_distribution[key] /= len(symbols)

    return symbol_distribution

class node():
    def __init__(self, pr=None, left=None, right=None, data=None):
        self.pr = pr
        self.data = data
        self.left = left
        self.right = right

def build_huff_tree(nodes):
    if len(nodes) == 1:
        return nodes[0]
    nodes = sorted(nodes, key=lambda n: n.pr)
    left = deepcopy(nodes[0])
    right = deepcopy(nodes[1])
    left.step = '0'
    right.step = '1'
    parent = node(left=left, right=right, data=str(left.data)+str(right.data), pr=left.pr+right.pr)
    nodes.append(parent)
    nodes.pop(0)
    nodes.pop(0)
    return build_huff_tree(nodes)

def path_to_key(key, root, path=''):
    if not root.right and not root.left:
        return path
    if str(key) in str(root.right.data):
        return path_to_key(key, root.right, path + root.right.step)
    if str(key) in str(root.left.data):
        return path_to_key(key, root.left, path + root.left.step)

def build_huffman_book(dist: dict):
    nodes = []
    for key, value in dist.items():
            nodes.append(node(data=key, pr=value))
    huff_root = build_huff_tree(nodes)
    book = {}
    for key, val in dist.items():
        book[key] = path_to_key(key, huff_root)
    return book

def create_bitstream():
    bitstream = ''
    amp_idx = 0
    for symbol in symbols:
        bitstream += huffman_book[symbol]
        if symbol[1] != 0:
            bitstream += str(amps[amp_idx])
            amp_idx += 1
    return bitstream

def avg_bitrate(book: dict, dist: dict):
    br = 0
    for i, code in enumerate(book.values()):
        br += len(code) * list(dist.values())[i]
    return br

def decode(bitstream: str, codebook: dict):
    symbols = []
    amps = []
    while bitstream:
        i=0
        curr_word = bitstream[i]
        while curr_word not in codebook:
            i += 1 
            try:
                curr_word += bitstream[i]
            except:
                raise RuntimeError
        symbols.append(codebook[curr_word])
        bitstream = bitstream[len(curr_word):]
        if codebook[curr_word][1] != 0:
            amps.append(bitstream[:codebook[curr_word][1]])
            bitstream = bitstream[codebook[curr_word][1]:]
    return symbols, amps

def inverse_amp(group, amp_symbol):
    if amp_symbol[0] == '0':
        return int(amp_symbol, 2) - amp_table[group]
    else:
        return int(amp_symbol, 2)

def subband_reconstruction(): # inverse run, size amplitude
    construction = np.zeros((512,512))
    curr_row = 0
    row_idx = 0
    amp_idx = 0
    for i, symbol in enumerate(newsymbols):
        if symbol == (0,0):
            curr_row += 1
            row_idx = 0
            continue
        row_idx += symbol[0] # preceding of zeros
        if row_idx >= 512:
            curr_row += 1
            row_idx = 0
        if symbol[1] != 0:
            construction[curr_row][row_idx] = inverse_amp(symbol[1], newamps[amp_idx])
            amp_idx += 1
        row_idx += 1
    return construction

img = cv2.imread('lena.jpg',0)
img = img.astype(np.double)


""" FORWARD DWT """
decomp_level = 4
x_dim = img.shape[0]
y_dim = img.shape[1]
max_run = 15
transformed = forwardDWT(img)
plt.imshow(transformed)
plt.show()

""" UNIFORM QUANTIZATION """ 
q = 16
quantized = np.round(np.divide(img, q))

""" RUN SIZE AMPLITUDE ENCODING """
symbols = []
amps = []
run_size(quantized)

""" HUFFMAN ENCODING AND BITSTREAM FORMATION """
sym_dist = symbol_dist(symbols)
huffman_book = build_huffman_book(sym_dist)
bitstream = create_bitstream()
bitrate = avg_bitrate(huffman_book, sym_dist)
print(bitrate)
""" HUFFMAN DECODING """
reverse_book = {v:k for k,v in huffman_book.items()}
newsymbols, newamps = decode(bitstream, reverse_book)

""" RUN SIZE AMPLITUDE DECODING """
constructed = subband_reconstruction()

""" INVERSE QUANTIZATION """ 
dequantized = np.multiply(constructed, q)

""" INVERSE DWT """ 
x_hat = inverseDWT(dequantized)

plt.imshow(x_hat)
plt.show()
mse = (np.square(img - x_hat)).mean(axis=None)
psnr = 10 * math.log10((255**2)/mse)
print(psnr)

