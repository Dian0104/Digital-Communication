import cmath as cm
import numpy as np
import math as mt

from typing import List, Tuple, Union

#########################################################################
#                  Type definitions and constants                       #
#########################################################################

BIT_SEQUENCE_TYPE = List[int]
SYMBOL_SEQUENCE_TYPE = List[complex]
BIT_TO_SYMBOL_MAP_TYPE = List[ List[ Union[ complex, List[int] ] ] ]
GENERATOR_MATRIX_TYPE = List[ List[int] ]
RANDOM_VALUES_SYMBOLS_TYPE = List[ List[float] ]
RANDOM_VALUES_RUN_TYPE = List[ List[ List[float] ] ]
SER_TYPE = List[float]
BER_TYPE = List[float]

MAP_BPSK : BIT_TO_SYMBOL_MAP_TYPE = [
    [(-1 + 0j), [0]],
    [(1 + 0j), [1]],
]

MAP_4QAM : BIT_TO_SYMBOL_MAP_TYPE = [
    [( 1 + 1j)/cm.sqrt(2), [0, 0]],
    [(-1 + 1j)/cm.sqrt(2), [0, 1]],
    [(-1 - 1j)/cm.sqrt(2), [1, 1]],
    [( 1 - 1j)/cm.sqrt(2), [1, 0]],
]

MAP_8PSK : BIT_TO_SYMBOL_MAP_TYPE = [
    [cm.rect(1, 0*cm.pi/4),    [1,1,1]],
    [cm.rect(1, 1*cm.pi/4),    [1,1,0]],
    [cm.rect(1, 2*cm.pi/4),    [0,1,0]],
    [cm.rect(1, 3*cm.pi/4),    [0,1,1]],
    [cm.rect(1, 4*cm.pi/4),    [0,0,1]],
    [cm.rect(1, 5*cm.pi/4),    [0,0,0]],
    [cm.rect(1, 6*cm.pi/4),    [1,0,0]],
    [cm.rect(1, 7*cm.pi/4),    [1,0,1]]
]

#########################################################################
#                  PLAYGROUND: Test your code here                      #
#########################################################################

def evaluate():

    G = [
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0],
    ]
    codeword = linear_block_codes_encode([1, 0, 0, 1, 0, 1, 1, 1], G)

    Q1_en = [ 1 ,0 ,0 ,1 ,0 ,1 ,1, 1, 1, 0, 1, 0, 1, 1 ]

    if codeword== Q1_en:
        print("Q1_en success")

    Q1_de = [1, 0, 0, 1, 0, 1, 1, 1]
    bits = linear_block_codes_decode(codeword, G)

    if Q1_de == bits:
        print("Q1_de success")
    
    print(codeword)
    print(bits)

    bits = [1, 1, 0, 1]
    print(bits)

    cdwrds = convolutional_codes_encode(bits)
    print(cdwrds)
    
    cdwrds = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0,1]
    bits = convolutional_codes_decode(cdwrds)
    print(bits)

    return

#########################################################################
#                  OPTIONAL: WILL NOT BE EVALUATED                      #
#########################################################################

def bit_to_symbol(bit_sequence: BIT_SEQUENCE_TYPE, bit_to_symbol_map: BIT_TO_SYMBOL_MAP_TYPE) -> SYMBOL_SEQUENCE_TYPE :
    """
    Converts a sequence of bits to a sequence of symbols using the bit to symbol map.

    parameters:
        bit_sequence -> type <class 'list'> : List containing int items which represents the bits for example: [0, 1, 1, 1, 0, 0, 1, 1]   
        bit_to_symbol_map -> type <class 'list'> : A list containing lists. Each list entry contains a complex number and a bit sequence, representing the symbol, and the corresponding bit sequence that maps to it.
                             the maps that will be used are given as MAP_BPSK, MAP_4QAM and MAP_8PSK
          Example:
            [
                [ 1+1j,    [0, 0] ],
                [ -1+1j,   [0, 1] ],
                [ -1-1j,   [1, 1] ],
                [ 1-1j,    [1, 0] ]
            ]
    returns:
        symbol_sequence -> type <class 'list'> : List containing complex items (<class "complex">) which represents the symbols for example: [1+1j, 2+2j, -0.66-0.25j]
    """

    symbol_sequence = []
    bits_per_symbol = len(bit_to_symbol_map[0][1])

    for index in range(0, len(bit_sequence), bits_per_symbol):
        bit_chunk = bit_sequence[index:index + bits_per_symbol]
        for symbol, bit_map in bit_to_symbol_map:
            if bit_chunk == bit_map:
                symbol_sequence.append(symbol)
                break

    return symbol_sequence

def symbol_to_bit(symbol_sequence, bit_to_symbol_map: BIT_TO_SYMBOL_MAP_TYPE) -> BIT_SEQUENCE_TYPE:
    """
    Returns a sequence of bits that corresponds to the provided sequence of symbols containing noise using the bit to symbol map that respresent the modulation scheme and the euclidean distance

    parameters:
        symbol_sequence -> type <class 'list'> : List containing complex items (<class "complex">) which represents the symbols for example: [1+1j, 2+2j]
        bit_to_symbol_map -> type <class 'list'> : A list containing lists. Each list entry contains a complex number and a bit sequence, representing the symbol, and the corresponding bit sequence that maps to it.
                             the maps that will be used are given as MAP_BPSK, MAP_4QAM and MAP_8PSK
          Example:
            [
                [ 1+1j,    [0, 0] ],
                [ -1+1j,   [0, 1] ],
                [ -1-1j,   [1, 1] ],
                [ 1-1j,    [1, 0] ]
            ]
        
    returns:
        bit_sequence -> type <class 'list'> : List containing int items which represents the bits for example: [0, 1, 1] 

    """

    bit_sequence = []

    for symbol in symbol_sequence:
        distances = [abs(symbol - mapped_symbol) ** 2 for mapped_symbol, _ in bit_to_symbol_map]
        closest_symbol_index = distances.index(min(distances))
        bit_sequence.extend(bit_to_symbol_map[closest_symbol_index][1])

    return bit_sequence

#########################################################################
#                   Question 1: Linear Block Codes                      #
#########################################################################

def linear_block_codes_encode(bit_sequence: BIT_SEQUENCE_TYPE, generator_matrix: GENERATOR_MATRIX_TYPE) -> BIT_SEQUENCE_TYPE:
    """"
    Takes the given sequence of bits and encodes it using linear block coding and the generator matrix. The function returns the encoded sequence. 
    
    The sequence of bits will match the size of the generator matrix.
    
    parameters:
        bit_sequence -> type <class 'list'> : List containing int items which represents the bits for example: [0, 1, 1]
        generator_matrix -> type <class 'list'> : A list containing lists, making a 2D array representing a matrix. The first index refers to the row and the second index refers to the column.
          Example (example 2 from lecture notes ):
            G = [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0],
            ]
        
    returns:
        codeword_sequence -> type <class 'list'> : List containing int items which represents the bits for example: [0, 1, 1]
    """

    encoded_sequence = []
    
   
    for j in range(len(generator_matrix[0])):  
        total = 0
        for i in range(len(bit_sequence)):    
            total += bit_sequence[i] * generator_matrix[i][j]
        encoded_sequence.append(total % 2)    
    
    return encoded_sequence

def linear_block_codes_decode(codeword_sequence: BIT_SEQUENCE_TYPE, generator_matrix: GENERATOR_MATRIX_TYPE) -> BIT_SEQUENCE_TYPE:
    
    """
    Takes the given sequence of bits (which may contain errors) and decodes it using linear block coding and the generator matrix, performing error correction coding. 
    The function returns the decoded sequence. If the method can not find and correct the errors, then return the codeword_sequence.  
    
    The sequence of bits will match the size of the generator matrix.
    
    parameters:
        codeword_sequence -> type <class 'list'> : List containing int items which represents the bits for example: [0, 1, 1]
        generator_matrix -> type <class 'list'> : A list containing lists, making a 2D array representing a matrix. The first index refers to the row and the second index refers to the column.
          Example (example 2 from lecture notes ):
            G = [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0],
            ]
        
    returns:
        bit_sequence -> type <class 'list'> : List containing int items which represents the bits for example: [0, 1, 1]
    """

    total_bits = len(generator_matrix[0])
    original_bits = len(generator_matrix)

    
    codeword_array = np.array(codeword_sequence)
    generator_matrix_array = np.array(generator_matrix)

    
    parity_part = generator_matrix_array[:, -(total_bits - original_bits):].T
    H_matrix = np.hstack((parity_part, np.eye(total_bits - original_bits)))
    
   
    syndrome = (codeword_array @ H_matrix.T) % 2

   
    error_indices = []
    for idx, row in enumerate(H_matrix.T):
        if np.array_equal(row, syndrome):
            error_indices.append(idx)
            break

   
    if not error_indices:
        for idx1, row1 in enumerate(H_matrix.T):
            if error_indices:
                break
            for idx2, row2 in enumerate(H_matrix.T):
                combined = (row1 + row2) % 2
                if np.array_equal(combined, syndrome):
                    error_indices.extend([idx1, idx2])
                    break

    
    for error_idx in error_indices:
        codeword_array[error_idx] = (codeword_array[error_idx] + 1) % 2

   
    return codeword_array[:original_bits].tolist()


def linear_block_codes_encode_long_sequence(bit_sequence: BIT_SEQUENCE_TYPE, generator_matrix: GENERATOR_MATRIX_TYPE) -> BIT_SEQUENCE_TYPE:
    """
    Takes the given sequence of bits and encodes it using linear block coding and the generator matrix. The function returns the encoded sequence. 
    
    The length of the bit_sequence is not going to match the generator matrix length, thus the bit sequence needs to be divided into smaller sequences, encoded using
    linear_block_codes_encode, and combined into a single larger bit sequence.
    
    parameters:
        bit_sequence -> type <class 'list'> : List containing int items which represents the bits for example: [0, 1, 1]
        generator_matrix -> type <class 'list'> : A list containing lists, making a 2D array representing a matrix. The first index refers to the row and the second index refers to the column.
          Example (example 2 from lecture notes ):
            G = [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0],
            ]
        
    returns:
        codeword_sequence -> type <class 'list'> : List containing int items which represents the bits for example: [0, 1, 1]
    """

    encoded_blocks = []
    
    
    for start_idx in range(0, len(bit_sequence), len(generator_matrix)):
        block = bit_sequence[start_idx:start_idx + len(generator_matrix)]
        encoded_blocks.extend(linear_block_codes_encode(block, generator_matrix))
    
    return encoded_blocks

def linear_block_codes_decode_long_sequence(codeword_sequence: BIT_SEQUENCE_TYPE, generator_matrix: GENERATOR_MATRIX_TYPE) -> BIT_SEQUENCE_TYPE:
    """
    Takes the given sequence of bits (which may contain errors) and decodes it using linear block coding and the generator matrix, performing error correction coding. 
    The function returns the decoded sequence. 
    
    The sequence of bits will not match the size of the generator matrix, it should thus be broken up into smaller sequences, decoded using linear_block_codes_decoding
    and combined to form a single decoded bit sequence.
    
    parameters:
        codeword_sequence -> type <class 'list'> : List containing int items which represents the bits for example: [0, 1, 1]
        generator_matrix -> type <class 'list'> : A list containing lists, making a 2D array representing a matrix. The first index refers to the row and the second index refers to the column.
          Example (example 2 from lecture notes ):
            G = [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0],
            ]
        
    returns:
        bit_sequence -> type <class 'list'> : List containing int items which represents the bits for example: [0, 1, 1]
    """

    decoded_bits = []

    for start_idx in range(0, len(codeword_sequence), len(generator_matrix[0])):
        block = codeword_sequence[start_idx:start_idx + len(generator_matrix[0])]
        decoded_bits.extend(linear_block_codes_decode(block, generator_matrix))

    return decoded_bits

#########################################################################
#                   Question 2: Convolutional Codes                     #
#########################################################################

def convolutional_codes_encode(bit_sequence: BIT_SEQUENCE_TYPE) -> BIT_SEQUENCE_TYPE:
    """
    Takes the given sequence of bits and encodes it using convolutional codes. The function returns the encoded sequence.
    The parameters for the encoder are provided in the practical guide.
    The sequence of bits can be any length.
    
    parameters:
        bit_sequence -> type <class 'list'> : List containing int items which represents the bits for example: [0, 1, 1]
        
    returns:
        codeword_sequence -> type <class 'list'> : List containing int items which represents the bits for example: [0, 1, 1]
    """
    
    K = 3
    g1 = np.array([1, 0, 0])
    g2 = np.array([1, 1, 0])
    g3 = np.array([1, 0, 1])

    # Append K-1 zeros to flush out bits from the shift register
    bit_sequence = bit_sequence + [0] * (K - 1)

    # Initialize the shift register
    SR = np.array([0] * K)
    encoded_sequence = []

    # Convolutional encoding process
    for bit in bit_sequence:
        # Shift the register and insert the new bit
        SR[1:] = SR[:-1]
        SR[0] = bit

        # Calculate output bits using generator polynomials
        c1 = int((SR @ g1) % 2)
        c2 = int((SR @ g2) % 2)
        c3 = int((SR @ g3) % 2)

        # Append encoded bits
        encoded_sequence.extend([c1, c2, c3])

    return encoded_sequence[:len(encoded_sequence)-K*2]

def convolutional_codes_decode(codeword_sequence: BIT_SEQUENCE_TYPE) -> BIT_SEQUENCE_TYPE:
    """
    Takes the given sequence of bits (which may contain errors) and decodes it using convolutional codes, performing error correction coding. 
    The parameters for the encoder are provided in the practical guide.
    The sequence of bits can be any length. 

    NOTE - Assume that zeros was appended to the original sequence before it was encoded and passed to this function. 
    
    parameters:
        codeword_sequence -> type <class 'list'> : List containing int items which represents the bits for example: [0, 1, 1]
        
    returns:
        bit_sequence -> type <class 'list'> : List containing int items which represents the bits for example: [0, 1, 1]
    """

    trellis_length = int(len(codeword_sequence) / 3) + 1
    L = 3

    def getCodeword(state, prev_state):
        ct = [[0],
              [0, 1],
              [0, 2]]

        window = [state[0]] + prev_state
        codeword = []

        for indices in ct:
            bit_sum = 0
            for index in indices:
                bit_sum += window[index]
            codeword.append(bit_sum % 2)
        return codeword

    def getNextStates(state):
        states = [
            [0, 2],
            [0, 2],
            [1, 3],
            [1, 3]
        ]
        return states[state]

    def getStateBits(state):
        stateBits = [[0, 0],
                     [0, 1],
                     [1, 0],
                     [1, 1]]
        return stateBits[state]

    def getSymbol(bit):
        symbols = [-1, 1]
        return symbols[bit]

    class Node:
        

        def getDelta(self):
            if self.delta is None:
                return 0
            else:
                return self.delta

        def __init__(self, t, state):
            self.t = t
            self.children = []
            self.parents = []
            self.delta = None
            self.nodePath = []
            self.state = state

    def getNodes(t):
        if t == 0 or t == trellis_length - 1:
            return [Node(t, 0)]
        if t == 1:
            return [Node(t, 0), Node(t, 2)]
        if t == trellis_length - 2:
            return [Node(t, 0), Node(t, 1)]
        return [Node(t, s) for s in range(4)]

    trellis = []

    def createTrellis():
        for t in range(trellis_length):
            trellis.append(getNodes(t))

        # create paths in the trellis
        for t in range(trellis_length - 1):
            nodes = trellis[t]
            nextNodes = trellis[t + 1]
            for node in nodes:
                nextStates = getNextStates(node.state)
                for nextNode in nextNodes:
                    if nextNode.state in nextStates:
                        node.children.append(nextNode)
                        nextNode.parents.append(node)

        return trellis[0][0], trellis[trellis_length - 1][0]

    def calculate_deltas(node, path):
        if len(node.parents):
            # ensure node path is full
            path = path[:]
            path.append(node)
            while len(path) < L:
                path.insert(0, Node(path[0].t - 1, 0))

            # calculate node delta for node path
            delta = 0
            c = getCodeword(getStateBits(node.state), getStateBits(path[-2].state))
            c_est = codeword_sequence[L * (node.t - 1):L * node.t]
            for n in range(L):
                delta += mt.pow(abs(getSymbol(c_est[n]) - getSymbol(c[n])), 2)
            # add previous node delta to current delta
            delta += path[-2].getDelta()

            # compare new and current delta values
            if node.delta is not None:
                if delta < node.delta:
                    node.delta = delta
                    node.nodePath = path
                else:
                    return
            else:
                node.delta = delta
                node.nodePath = path

            for child in node.children:
                calculate_deltas(child, node.nodePath)
        else:
            for child in node.children:
                calculate_deltas(child, node.nodePath)

    def get_path(node):
        path = []
        for path_node in node.nodePath:
            path.append(getStateBits(path_node.state)[0])

        for i in range(L - 1):
            path.pop(0)
            path.pop()

        return path

  

    root, leaf = createTrellis()
    calculate_deltas(root, root.nodePath)
    bit_sequence = get_path(leaf)


    return bit_sequence

def convolutional_codes_encode_long_sequence(bit_sequence: BIT_SEQUENCE_TYPE) -> BIT_SEQUENCE_TYPE:
    """
    Takes the given sequence of bits and encodes it using convolutional codes. The function returns the encoded sequence. 
    The parameters for the encoder are provided in the practical guide.
    
    The sequence of bits should be broken up into smaller sequences, zeros should be appended to the end of each sequence, and then encoded using convolutional_codes_encode 
    to yield multiple sequences of length Nc = 300. All the encoded sequences should then be combined to form a single larger sequence.

    parameters:
        bit_sequence -> type <class 'list'> : List containing int items which represents the bits for example: [0, 1, 1]
        generator_matrix -> type <class 'list'> : A list containing lists, making a 2D array representing a matrix. The first index refers to the row and the second index refers to the column.
          Example (example 2 from lecture notes ):
            G = [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0],
            ]
        
    returns:
        codeword_sequence -> type <class 'list'> : List containing int items which represents the bits for example: [0, 1, 1]
    """

    def code(bit_sequence: BIT_SEQUENCE_TYPE):
        K = 3
        g1 = np.array([1, 0, 0])
        g2 = np.array([1, 1, 0])
        g3 = np.array([1, 0, 1])

        # Append K-1 zeros to flush out bits from the shift register
        bit_sequence = bit_sequence + [0] * (K - 1)

        # Initialize the shift register
        SR = np.array([0] * K)
        encoded_sequence = []

        # Convolutional encoding process
        for bit in bit_sequence:
            # Shift the register and insert the new bit
            SR[1:] = SR[:-1]
            SR[0] = bit

            # Calculate output bits using generator polynomials
            c1 = int((SR @ g1) % 2)
            c2 = int((SR @ g2) % 2)
            c3 = int((SR @ g3) % 2)

            # Append encoded bits
            encoded_sequence.extend([c1, c2, c3])

        return encoded_sequence

    codeword_sequence = []

    for start in range(0, len(bit_sequence), 98):
        bits = bit_sequence[start: start + 98]
        codeword_sequence += code(bits)

    return codeword_sequence

def convolutional_codes_decode_long_sequence(codeword_sequence: BIT_SEQUENCE_TYPE) -> BIT_SEQUENCE_TYPE:
    """
    Takes the given sequence of bits (which may contain errors) and decodes it using convolutional codes, performing error correction coding.  
    The parameters for the encoder are provided in the practical guide.
    
    The sequence will consist of multiple codewords sequences of length 300, which should be decoded using convolutional_codes_decode, 
    and then recombined into a single decoded sequence with the appended zeros removed.
    
    parameters:
        codeword_sequence -> type <class 'list'> : List containing int items which represents the bits for example: [0, 1, 1]
        
    returns:
        bit_sequence -> type <class 'list'> : List containing int items which represents the bits for example: [0, 1, 1]
    """

    bit_sequence = []

    for start in range(0, len(codeword_sequence), 300):
        sequence = codeword_sequence[start: start + 300]
        bit_sequence += convolutional_codes_decode(sequence)

    return bit_sequence

#########################################################################
#                   Question 3: AWGN and BER                            #
#########################################################################
def calculateSigma(snr, fbit, Rc):
    sigma = 10**(snr*0.1)
    sigma *= fbit
    sigma *= Rc
    sigma = np.sqrt(sigma)
    sigma = 1/sigma

    return sigma
def AWGN_Transmission(bit_sequence: BIT_SEQUENCE_TYPE, bit_to_symbol_map: BIT_TO_SYMBOL_MAP_TYPE, random_values_for_symbols: RANDOM_VALUES_SYMBOLS_TYPE, snr: float, transmission_rate: float) -> Tuple[SYMBOL_SEQUENCE_TYPE, BIT_SEQUENCE_TYPE]:
    """
    This function takes the given bit sequence, modulate it using the bit to symbol map, add noise to the symbol sequence as described in the practical guide, 
    and demodulate it back into a noisy bit sequence. The function returns this generated noisy bit sequence, along with the noisy symbol sequence with the added noise.
    
    NOTE - As with the previous practicals, BPSK uses a different equation
    
    parameters:
        bit_sequence -> type <class 'list'> : List containing int items which represents the bits for example: [0, 1, 1]
        bit_to_symbol_map -> type <class 'list'> : A list containing lists. Each list entry contains a complex number and a bit sequence, representing the symbol, and the corresponding bit sequence that maps to it.
                             the maps that will be used are given as MAP_BPSK, MAP_4QAM and MAP_8PSK
          Example:
            [
                [ 1+1j,    [0, 0] ],
                [ -1+1j,   [0, 1] ],
                [ -1-1j,   [1, 1] ],
                [ 1-1j,    [1, 0] ]
            ]
        random_values_for_symbols -> type <class 'list'> : List containing lists. Each entry is a list containing two values, which are random Gaussian zero mean unity variance values. 
                                                           The first index refers to the symbol in the sequence, and the second index refers to the real or imaginary kappa value.
          Example:
            [[1.24, 0.42], [-1.2, -0.3], [0, 1.23], [-0.3, 1.2]]
        snr -> type <class 'float'> : A float value which is the SNR that should be used when adding noise
        transmission_rate -> type <class 'flaot'> : A float value which is the transmission rate (Rc), which should be used when adding noise.

    returns:
        noisy_symbol_sequence -> type <class 'list'> : List containing complex items (<class "complex">) which represents the symbols for example: [1+1j, 2+2j]
        noisy_bit_sequence -> type <class 'list'> : List containing int items which represents the bits for example: [0, 1, 1]
    """

    def get_distance(z1, z2):
        a, b = z1.real, z1.imag
        c, d = z2.real, z2.imag
        distance = np.sqrt((c - a) ** 2 + (d - b) ** 2)
        return distance

    symbol_map = [bit_to_symbol_map[row][0] for row in range(len(bit_to_symbol_map))]
    bit_map = [bit_to_symbol_map[row][1] for row in range(len(bit_to_symbol_map))]

    symbol_sequence = bit_to_symbol(bit_sequence, bit_to_symbol_map)

    sigma = 1 / (np.sqrt(pow(10, snr / 10) * np.log2(len(bit_to_symbol_map)) * transmission_rate))

    is_bpsk = len(bit_map) == 2

    nk = []
    rk = []

    for i in range(len(random_values_for_symbols)):
        if is_bpsk:
            nk.append((random_values_for_symbols[i][0]))
        else:
            nk.append((random_values_for_symbols[i][0] + random_values_for_symbols[i][1] * 1j) / np.sqrt(2))
    # print(len(symbol_sequence))

    for i in range(len(symbol_sequence)):
        rk.append(symbol_sequence[i] + sigma * nk[i])

    noisy_symbol_sequence = []

    for noisy_symbol_index, noisy_symbol in enumerate(rk):
        distances = []
        for comparison_symbol in symbol_map:
            distances.append(get_distance(noisy_symbol, comparison_symbol))

        min_index = distances.index(min(distances))
        noisy_symbol_sequence.append(symbol_map[min_index])

    noisy_bit_sequence = symbol_to_bit(noisy_symbol_sequence, bit_to_symbol_map)

    return noisy_symbol_sequence, noisy_bit_sequence

def BER_linear_block_codes(bit_sequence: BIT_SEQUENCE_TYPE, generator_matrix: GENERATOR_MATRIX_TYPE, bit_to_symbol_map: BIT_TO_SYMBOL_MAP_TYPE, random_values_for_runs: RANDOM_VALUES_RUN_TYPE, snr_range: List[float]) -> List[float]:
    """
    This functions simulates the linear block codes method over a AWGN channel for different snr values. The snr_range argument, provides the snr values that should be used for
    each run. Each run encodes the long bit sequence using linear block codes, transmits it over the AWGN channel, decodes it, and then calculate the BER using the input bit sequence
    and the final bit sequence. This is repeated for each snr value, and the results for each run is stored in a list and returned. 
    
    parameters:
        bit_sequence -> type <class 'list'> : List containing int items which represents the bits for example: [0, 1, 1]
        bit_to_symbol_map -> type <class 'list'> : A list containing lists. Each list entry contains a complex number and a bit sequence, representing the symbol, and the corresponding bit sequence that maps to it.
                             the maps that will be used are given as MAP_BPSK, MAP_4QAM and MAP_8PSK
          Example:
            [
                [ 1+1j,    [0, 0] ],
                [ -1+1j,   [0, 1] ],
                [ -1-1j,   [1, 1] ],
                [ 1-1j,    [1, 0] ]
            ]
        random_values_for_symbols -> type <class 'list'> : List containing lists containing lists. Each entry is a list containing multiple lists with two values, which are random Gaussian zero mean unity variance values. 
                                                           The first index refers to run (corresponding with the snr value), 
                                                           The second index refers to the symbol within that run,
                                                           The third index refers to the real or imaginary kappa value
          Example:
            [
                [1.24, 0.42], [-1.2, -0.3], [0, 1.23], [-0.3, 1.2],
                [-0.3, 0.42], [-0.32, 0.42], [1.24, 1.23], [-0.3, 1.2],
                [1.24, 1.24], [-1.2, 0.42], [0, 1.23], [0.42, -0.3],
            ]
        snr_range -> type <class 'list'> : A list containing float values which are the SNR that should be used when adding noise during the different runs
        
    returns:
        BER_values -> type <class 'list'> : A list containing float values which are the BER results for the different runs, corresponding to the snr value in the snr_range.
    """

    BER_values = []
    Fbit = np.log2(len(bit_to_symbol_map))
    codewords = linear_block_codes_encode_long_sequence(bit_sequence, generator_matrix)
    for i,j in zip(snr_range, random_values_for_runs):
        (noisy_symbols, noisy_bits) = AWGN_Transmission(codewords, bit_to_symbol_map, j, i, 1/2)

        correct_bits_code = linear_block_codes_decode_long_sequence(noisy_bits,generator_matrix)

        BER_temp = 0

        for z,k in zip(correct_bits_code, bit_sequence):
            BER_temp += not(z == k)

        BER_temp = BER_temp/len(bit_sequence)
        BER_values.append(BER_temp)

    return BER_values

def BER_convolution_codes(bit_sequence: BIT_SEQUENCE_TYPE, bit_to_symbol_map: BIT_TO_SYMBOL_MAP_TYPE, random_values_for_runs: RANDOM_VALUES_RUN_TYPE, snr_range: List[float]) -> List[float]:
    """
    This functions simulates the convolutional codes method over a AWGN channel for different snr values. The snr_range argument, provides the snr values that should be used for
    each run. Each run encodes the long bit sequence using convolutional codes, transmits it over the AWGN channel, decodes it, and then calculate the BER using the input bit sequence
    and the final bit sequence. This is repeated for each snr value, and the results for each run is stored in a list and returned. 

    parameters:
        bit_sequence -> type <class 'list'> : List containing int items which represents the bits for example: [0, 1, 1]
        bit_to_symbol_map -> type <class 'list'> : A list containing lists. Each list entry contains a complex number and a bit sequence, representing the symbol, and the corresponding bit sequence that maps to it.
                             the maps that will be used are given as MAP_BPSK, MAP_4QAM and MAP_8PSK
          Example:
            [
                [ 1+1j,    [0, 0] ],
                [ -1+1j,   [0, 1] ],
                [ -1-1j,   [1, 1] ],
                [ 1-1j,    [1, 0] ]
            ]
        random_values_for_symbols -> type <class 'list'> : List containing lists containing lists. Each entry is a list containing multiple lists with two values, which are random Gaussian zero mean unity variance values. 
                                                           The first index refers to run (corresponding with the snr value), 
                                                           The second index refers to the symbol within that run,
                                                           The third index refers to the real or imaginary kappa value
          Example:
            [
                [1.24, 0.42], [-1.2, -0.3], [0, 1.23], [-0.3, 1.2],
                [-0.3, 0.42], [-0.32, 0.42], [1.24, 1.23], [-0.3, 1.2],
                [1.24, 1.24], [-1.2, 0.42], [0, 1.23], [0.42, -0.3],
            ]
        snr_range -> type <class 'list'> : A list containing float values which are the SNR that should be used when adding noise during the different runs
        
    returns:
        BER_values -> type <class 'list'> : A list containing float values which are the BER results for the different runs, corresponding to the snr value in the snr_range.
    """

    BER_values = []
    Fbit = np.log2(len(bit_to_symbol_map))
    codewords = convolutional_codes_encode_long_sequence(bit_sequence)
    for i,j in zip(snr_range, random_values_for_runs):

        (noisy_symbols, noisy_bits) = AWGN_Transmission(codewords, bit_to_symbol_map, j, i, 1/2)

        correct_bits_code = convolutional_codes_decode_long_sequence(noisy_bits)

        BER_temp = 0

        for z,k in zip(correct_bits_code, bit_sequence):
            BER_temp += not(z == k)

        BER_temp = BER_temp/len(bit_sequence)
        BER_values.append(BER_temp)

    return BER_values


######## DO NOT EDIT ########
if __name__ == "__main__" :
    evaluate()
#############################
