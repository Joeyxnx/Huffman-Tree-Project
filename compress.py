from __future__ import annotations

import time
from typing import Optional, Tuple, Dict
from huffman import HuffmanTree
from utils import *
# ====================
# Functions for compression

def build_frequency_dict(text: bytes) -> dict[int, int]:
    """ Return a dictionary which maps each of the bytes in <text> to its
    frequency.

    >>> d = build_frequency_dict(bytes([65, 66, 67, 66]))
    >>> d == {65: 1, 66: 2, 67: 1}
    True
    """
    new_freq_dict = {}
    for dict_bytes in text:
        if dict_bytes in new_freq_dict:
            new_freq_dict[dict_bytes] += 1
        else:
            new_freq_dict[dict_bytes] = 1
    return new_freq_dict


def build_huffman_tree(freq_dict: dict[int, int]) -> HuffmanTree:
    """ Return the Huffman tree corresponding to the frequency dictionary
    <freq_dict>.

    Precondition: freq_dict is not empty.

    >>> freq = {2: 6, 3: 4}
    >>> t = build_huffman_tree(freq)
    >>> result = HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    >>> t == result
    True
    >>> freq = {2: 6, 3: 4, 7: 5}
    >>> t = build_huffman_tree(freq)
    >>> result = HuffmanTree(None, HuffmanTree(2), \
                             HuffmanTree(None, HuffmanTree(3), HuffmanTree(7)))
    >>> t == result
    True
    >>> import random
    >>> symbol = random.randint(0,255)
    >>> freq = {symbol: 6}
    >>> t = build_huffman_tree(freq)
    >>> any_valid_byte_other_than_symbol = (symbol + 1) % 256
    >>> dummy_tree = HuffmanTree(any_valid_byte_other_than_symbol)
    >>> result = HuffmanTree(None, HuffmanTree(symbol), dummy_tree)
    >>> t.left == result.left or t.right == result.left
    True
    """
    new_list = []
    for obj in freq_dict:
        new_list.append((freq_dict[obj], HuffmanTree(obj)))

    if len(new_list) != 1:
        pass
    else:
        huff = HuffmanTree(None, new_list[0][1], new_list[0][1])
        return huff

    while 1 < len(new_list):
        new_list.sort(key=lambda x: x[
            0])  # https://stackoverflow.com/questions/36955553/
        # sorting-list-of-lists-by-the-first-element-of-each-sub-list#:~:text=
        # Use%20sorted%20function%20along%20with%20passing%20anonymous%
        # 20function4%2C%207%5D%2C%20%5B2%2C%2059%2C%208%5D%2C%20%5B3%2C%
        # 206%2C%209%5D%5D
        left_obj, right_obj = new_list.pop(0), new_list.pop(0)
        new_obj = (left_obj[0] + right_obj[0], HuffmanTree(left=left_obj[1],
                                                           right=right_obj[1]))
        new_list = new_list + [new_obj]
    return new_list[0][1]


def get_codes(tree: HuffmanTree) -> dict[int, str]:
    """ Return a dictionary which maps symbols from the Huffman tree <tree>
    to codes.

    >>> tree = HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    >>> d = get_codes(tree)
    >>> d == {3: "0", 2: "1"}
    True
    """
    dict_symbols = {}
    new_list = [(tree, '')]
    while new_list:
        node, symbol = new_list.pop()
        if isinstance(node.symbol, int):
            dict_symbols[node.symbol] = symbol
        else:
            new_list.append((node.left, symbol + '0'))
            new_list.append((node.right, symbol + '1'))
    return dict_symbols


def number_nodes(tree: HuffmanTree) -> None:
    """ Number internal nodes in <tree> according to postorder traversal. The
    numbering starts at 0.

    >>> left = HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    >>> right = HuffmanTree(None, HuffmanTree(9), HuffmanTree(10))
    >>> tree = HuffmanTree(None, left, right)
    >>> number_nodes(tree)
    >>> tree.left.number
    0
    >>> tree.right.number
    1
    >>> tree.number
    2
    """

    def _helper(huff_tree: HuffmanTree, count_nodes: int) -> int:
        if huff_tree.symbol is None and huff_tree.left and huff_tree.right:
            count_nodes = _helper(huff_tree.left, count_nodes)
            count_nodes = _helper(huff_tree.right, count_nodes)
            huff_tree.number = count_nodes
            return count_nodes + 1
        return count_nodes

    _helper(tree, 0)


def avg_length(tree: HuffmanTree, freq_dict: dict[int, int]) -> float:
    """ Return the average number of bits required per symbol, to compress the
    text made of the symbols and frequencies in <freq_dict>, using the Huffman
    tree <tree>.

    The average number of bits = the weighted sum of the length of each symbol
    (where the weights are given by the symbol's frequencies), divided by the
    total of all symbol frequencies.

    >>> freq = {3: 2, 2: 7, 9: 1}
    >>> left = HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    >>> right = HuffmanTree(9)
    >>> tree = HuffmanTree(None, left, right)
    >>> avg_length(tree, freq)  # (2*2 + 7*2 + 1*1) / (2 + 7 + 1)
    1.9
    """
    codes = get_codes(tree)
    bytes_result = sum(freq_dict[key] * len(codes[key]) for key in freq_dict)
    freq_result = sum(freq_dict.values())
    return bytes_result / freq_result


def compress_bytes(text: bytes, codes: dict[int, str]) -> bytes:
    """ Return the compressed form of <text>, using the mapping from <codes>
    for each symbol.

    >>> d = {0: "0", 1: "10", 2: "11"}
    >>> text = bytes([1, 2, 1, 0])
    >>> result = compress_bytes(text, d)
    >>> result == bytes([184])
    True
    >>> [byte_to_bits(byte) for byte in result]
    ['10111000']
    >>> text = bytes([1, 2, 1, 0, 2])
    >>> result = compress_bytes(text, d)
    >>> [byte_to_bits(byte) for byte in result]
    ['10111001', '10000000']
    """
    symbol_item = ''.join(codes[obj] for obj in text)
    length = (len(symbol_item) + 7) // 8
    list_of_bytes = [bits_to_byte(symbol_item[i * 8:(i + 1) * 8])
                     for i in range(length)]
    return bytes(list_of_bytes)


def tree_to_bytes(tree: HuffmanTree) -> bytes:
    """ Return a bytes representation of the Huffman tree <tree>.
    The representation should be based on the postorder traversal of the tree's
    internal nodes, starting from 0.

    Precondition: <tree> has its nodes numbered.

    >>> tree = HuffmanTree(None, HuffmanTree(3, None, None), \
    HuffmanTree(2, None, None))
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))
    [0, 3, 0, 2]
    >>> left = HuffmanTree(None, HuffmanTree(3, None, None), \
    HuffmanTree(2, None, None))
    >>> right = HuffmanTree(5)
    >>> tree = HuffmanTree(None, left, right)
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))
    [0, 3, 0, 2, 1, 0, 0, 5]
    >>> tree = build_huffman_tree(build_frequency_dict(b"helloworld"))
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))\
            #doctest: +NORMALIZE_WHITESPACE
    [0, 104, 0, 101, 0, 119, 0, 114, 1, 0, 1, 1, 0, 100, 0, 111, 0, 108,\
    1, 3, 1, 2, 1, 4]
    """
    new_list = []

    def _tree_to_bytes_helper(huff_tree: HuffmanTree) -> [int]:
        """
        helper function for tree to bytes
        """
        for node in (huff_tree.left, huff_tree.right):
            if node:
                if node.is_leaf():
                    new_list.extend([0, node.symbol])
                else:
                    new_list.extend([1, node.number])

    def _post_order_visit(node_tree: HuffmanTree) -> \
            Optional[
                Tuple[
                    Optional[Tuple[int, int]],
                    Optional[Tuple[int, int]],
                    int]]:
        """
        visit tree in post order
        """
        if node_tree and not node_tree.symbol:
            result = _post_order_visit(node_tree.left), \
                _post_order_visit(node_tree.right), \
                _tree_to_bytes_helper(node_tree)
            return result
        return None
    _post_order_visit(tree)
    return bytes(new_list)


def compress_file(in_file: str, out_file: str) -> None:
    """ Compress contents of the file <in_file> and store results in <out_file>.
    Both <in_file> and <out_file> are string objects representing the names of
    the input and output files.

    Precondition: The contents of the file <in_file> are not empty.
    """
    with open(in_file, "rb") as f1:
        text = f1.read()
    freq = build_frequency_dict(text)
    tree = build_huffman_tree(freq)
    codes = get_codes(tree)
    number_nodes(tree)
    print("Bits per symbol:", avg_length(tree, freq))
    result = (tree.num_nodes_to_bytes() + tree_to_bytes(tree)
              + int32_to_bytes(len(text)))
    result += compress_bytes(text, codes)
    with open(out_file, "wb") as f2:
        f2.write(result)


# ====================
# Functions for decompression

def generate_tree_general(node_lst: list[ReadNode],
                          root_index: int) -> HuffmanTree:
    """ Return the Huffman tree corresponding to node_lst[root_index].
    The function assumes nothing about the order of the tree nodes in the list.

    >>> lst = [ReadNode(0, 5, 0, 7), ReadNode(0, 10, 0, 12), \
    ReadNode(1, 1, 1, 0)]
    >>> generate_tree_general(lst, 2)
    HuffmanTree(None, HuffmanTree(None, HuffmanTree(10, None, None), \
HuffmanTree(12, None, None)), \
HuffmanTree(None, HuffmanTree(5, None, None), HuffmanTree(7, None, None)))
    """
    value_root = node_lst[root_index]
    if value_root.l_type != 0:
        left_side = generate_tree_general(node_lst, value_root.l_data)
    else:
        left_side = HuffmanTree(value_root.l_data)
    if value_root.r_type != 0:
        right_side = generate_tree_general(node_lst, value_root.r_data)
    else:
        right_side = HuffmanTree(value_root.r_data)
    result = HuffmanTree(None, left_side, right_side)
    return result


def generate_tree_postorder(node_lst: list[ReadNode],
                            root_index: int) -> HuffmanTree:
    """ Return the Huffman tree corresponding to node_lst[root_index].
    The function assumes that the list represents a tree in postorder.

    >>> lst = [ReadNode(0, 5, 0, 7), ReadNode(0, 10, 0, 12), \
    ReadNode(1, 0, 1, 0)]
    >>> generate_tree_postorder(lst, 2)
    HuffmanTree(None, HuffmanTree(None, HuffmanTree(5, None, None), \
HuffmanTree(7, None, None)), \
HuffmanTree(None, HuffmanTree(10, None, None), HuffmanTree(12, None, None)))
    """
    curr_node = None
    if node_lst[root_index].l_type == 0:
        if node_lst[root_index].r_type == 0:
            curr_node = HuffmanTree(None,
                                    HuffmanTree(node_lst[root_index].l_data),
                                    HuffmanTree(node_lst[root_index].r_data))
    elif node_lst[root_index].l_type == 0:
        if node_lst[root_index].r_type == 1:
            curr_node = HuffmanTree(None,
                                    HuffmanTree(node_lst[root_index].l_data),
                                    generate_tree_postorder(node_lst,
                                                            (root_index - 1)))
    elif node_lst[root_index].r_type == 0:
        if node_lst[root_index].l_type == 1:
            curr_node = HuffmanTree(None,
                                    generate_tree_postorder(node_lst,
                                                            (root_index - 1)),
                                    HuffmanTree(node_lst[root_index].r_data))
    else:
        right_side = generate_tree_postorder(node_lst, (root_index - 1))
        number_nodes(right_side)
        i_new = (root_index - 1) - (right_side.number + 1)
        curr_node = HuffmanTree(None,
                                generate_tree_postorder(node_lst, i_new),
                                right_side)
    if curr_node is not None:
        number_nodes(curr_node)
    return curr_node


def decompress_bytes(tree: HuffmanTree, text: bytes, size: int) -> bytes:
    """ Use Huffman tree <tree> to decompress <size> bytes from <text>.

    >>> tree = build_huffman_tree(build_frequency_dict(b'helloworld'))
    >>> number_nodes(tree)
    >>> decompress_bytes(tree, \
             compress_bytes(b'helloworld', get_codes(tree)), len(b'helloworld'))
    b'helloworld'
    """
    bytes_var = ''

    def _decompress_helper(decompress: Dict[int, Tuple[int, int]]) -> \
            Dict[Tuple[int, int], int]:
        """
        Helper function to decompress bytes
        """
        result = dict((value, key) for key, value in decompress.items())
        return result

    codes = _decompress_helper(get_codes(tree))
    for items in list(text):
        bytes_var += byte_to_bits(items)
    decompress_lst = []
    str_text = ''
    counter = 0
    while counter < len(bytes_var):
        str_text += bytes_var[counter]
        if str_text in codes:
            decompress_lst.append(codes[str_text])
            str_text = ''
            if len(decompress_lst) == size:
                break
        counter += 1

    return bytes(decompress_lst)


def decompress_file(in_file: str, out_file: str) -> None:
    """ Decompress contents of <in_file> and store results in <out_file>.
    Both <in_file> and <out_file> are string objects representing the names of
    the input and output files.

    Precondition: The contents of the file <in_file> are not empty.
    """
    with open(in_file, "rb") as f:
        num_nodes = f.read(1)[0]
        buf = f.read(num_nodes * 4)
        node_lst = bytes_to_nodes(buf)
        # use generate_tree_general or generate_tree_postorder here
        tree = generate_tree_general(node_lst, num_nodes - 1)
        size = bytes_to_int(f.read(4))
        with open(out_file, "wb") as g:
            text = f.read()
            g.write(decompress_bytes(tree, text, size))


# ====================
# Other functions

def improve_tree(tree: HuffmanTree, freq_dict: dict[int, int]) -> None:
    """ Improve the tree <tree> as much as possible, without changing its shape,
    by swapping nodes. The improvements are with respect to the dictionary of
    symbol frequencies <freq_dict>.

    >>> left = HuffmanTree(None, HuffmanTree(99, None, None), \
    HuffmanTree(100, None, None))
    >>> right = HuffmanTree(None, HuffmanTree(101, None, None), \
    HuffmanTree(None, HuffmanTree(97, None, None), HuffmanTree(98, None, None)))
    >>> tree = HuffmanTree(None, left, right)
    >>> freq = {97: 26, 98: 23, 99: 20, 100: 16, 101: 15}
    >>> avg_length(tree, freq)
    2.49
    >>> improve_tree(tree, freq)
    >>> avg_length(tree, freq)
    2.31
    """
    new_tuple = []
    for byte, freq in freq_dict.items():
        new_tuple.append((byte, freq))
    new_tuple.sort(key=lambda x: x[1])  # https://stackoverflow.com/questions
    # /36955553/sorting-list-of-lists-by-the-first-element-of-each-sub-list#:~:
    # text=Use%20sorted%20function%20along%20with%20passing%20anonymous%
    # 20function4%2C%207%5D%2C%20%5B2%2C%2059%2C%208%5D%2C%20%5B3%2C%
    # 206%2C%209%5D%5D
    traversal = [tree]
    while traversal:
        node = traversal.pop(0)
        if node.is_leaf():
            node.symbol = new_tuple.pop()[0]
        if node.left:
            traversal.append(node.left)
        if node.right:
            traversal.append(node.right)


if __name__ == "__main__":
    import doctest

    doctest.testmod()

    import python_ta

    python_ta.check_all(config={
        'allowed-io': ['compress_file', 'decompress_file'],
        'allowed-import-modules': [
            'python_ta', 'doctest', 'typing', '__future__',
            'time', 'utils', 'huffman', 'random'
        ],
        'disable': ['W0401']
    })

    mode = input(
        "Press c to compress, d to decompress, or other key to exit: ")
    if mode == "c":
        fname = input("File to compress: ")
        start = time.time()
        compress_file(fname, fname + ".huf")
        print(f"Compressed {fname} in {time.time() - start} seconds.")
    elif mode == "d":
        fname = input("File to decompress: ")
        start = time.time()
        decompress_file(fname, fname + ".orig")
        print(f"Decompressed {fname} in {time.time() - start} seconds.")
