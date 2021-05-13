import itertools, numpy as np
from copy import deepcopy

def prefix_decode(bitstream: str, codebook: dict, out=False) -> bool:
    """ dont really need this but the pseudocode was in the book """
    res = ''
    while bitstream:
        i=0
        curr_word = bitstream[i]
        while curr_word not in codebook:
            i += 1 
            try:
                curr_word += bitstream[i]
            except:
                return False
        res += codebook[curr_word]
        bitstream = bitstream[len(curr_word):]
    if out:
        print(res)
    return True

def valid_and_unique(bitstream: str, codebook: dict):
    if not bitstream:
        return True
    ways_to_decode = 0
    for key in codebook.keys():
        if bitstream.startswith(key):
            ways_to_decode += valid_and_unique(bitstream[len(key):], codebook)
    return ways_to_decode == 1

def test_permutations(book: dict, prefix_encoding=False):
    streams = list(itertools.permutations(book.keys()))
    if prefix_encoding:
        streams = [prefix_decode(''.join(stream), book) for stream in streams]
        if all(streams):
            print('{} is a valid prefix encoding'.format(book))
        else:
            print('{} is NOT a valid prefix encoding'.format(book))
    else:
        streams = [valid_and_unique(''.join(stream), book) for stream in streams]
        if all(streams):
            print('{} is uniquely decodable'.format(book))
        else:
            print('{} is NOT uniquely decodable'.format(book))

def test_kraft(book: dict):
    sum = 0
    for key in book.keys():
        sum += (1 / 2**len(key))
    if sum <= 1:
        print('{} satisfies the kraft inequality with sum {}'.format(book, sum))

def one_dim_entropy(px: list):
    Hx = 0
    for p in px:
        Hx += p * np.log2(1/p)
    return Hx

prefix_book = {
    '10': 'a',
    '11': 'b',
    '011': 'c',
    '010': 'd',
    '001': 'e'
}
test_kraft(prefix_book)
test_permutations(prefix_book, prefix_encoding=True)

bad_book = {
    '10': 'a',
    '11': 'b',
    '1101': 'c',
    '00': 'd',
    '1011': 'e'
}
test_permutations(bad_book)

unique_notprefix = {
    '00': 'a',
    '10': 'b',
    '11': 'c',
    '1001': 'd',
    '1101': 'e'
}
test_permutations(unique_notprefix)
# we can tell with our eyes but,
test_permutations(unique_notprefix, prefix_encoding=True)
# will verify this is not a prefix encoding

dist = {
    'a': 0.1, 
    'b': 0.12,
    'c': 0.08,
    'd': 0.15,
    'e': 0.05, 
    'f': 0.05, 
    'g': 0.2, 
    'h': 0.10, 
    'i': 0.15,
}
Hx = one_dim_entropy(dist.values())
print('The entropy of {} is {}'.format(list(dist.values()), Hx))

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
    parent = node(left=left, right=right, data=left.data+right.data, pr=left.pr+right.pr)
    nodes.append(parent)
    nodes.pop(0)
    nodes.pop(0)
    return build_huff_tree(nodes)

def path_to_key(key, root, path=''):
    if not root.right and not root.left:
        return path
    if key in root.right.data:
        return path_to_key(key, root.right, path + root.right.step)
    if key in root.left.data:
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

huffman_book = build_huffman_book(dist)
print('huffman encoding for {} is \n{}'.format(dist, huffman_book))
test_permutations(huffman_book, True)
reverse_book = {v:k for k,v in huffman_book.items()}
test_kraft(reverse_book)

from chardet import detect
def get_encoding_type(file):
    with open(file, 'rb') as f:
        rawdata = f.read()
    return detect(rawdata)['encoding']

filename = 'state.txt'
f = open(filename, 'r', encoding = get_encoding_type(filename))
contents = f.read()
counts = {}
for char in contents:
    if char not in counts:
        counts[char] = 0
    counts[char] += 1

text_dist = {k: v/len(contents) for k, v in counts.items()}
hbook = build_huffman_book(text_dist)

def encode_text(text, book):
    out = ''
    for char in text:
        out += book[char]
    return out
encoded_text = encode_text(contents, hbook)

def avg_bitrate(book: dict, dist: dict):
    br = 0
    for i, code in enumerate(book.values()):
        br += len(code) * list(dist.values())[i]
    return br
    
print('average bitrate of my text file encoding is {}'.format(avg_bitrate(hbook, text_dist)))
print('entropy for the text file was found to be {}'.format(one_dim_entropy(list(text_dist.values()))))

# decode and print the original message using our prefix decode from before
decode_book = {v:k for k,v in hbook.items()} #switch the keys and values for decoding
if prefix_decode(encoded_text, decode_book, out=False): #set to true to print the text file
    print(' the text file encoding is a valid prefix code')

print(hbook)