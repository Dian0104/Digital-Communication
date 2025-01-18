import cmath as cm
from typing import List, Union, Tuple
import math as m
import numpy as np

############################# Defining Types ############################

BIT_SEQUENCE_TYPE = List[int]
SYMBOL_SEQUENCE_TYPE = List[complex]
BIT_TO_SYMBOL_MAP_TYPE = List[ List[ Union[ complex, List[int] ] ] ]
SYMBOL_BLOCKS_TYPE = List[ List[complex] ]
CHANNEL_IMPULSE_RESPONSE_TYPE = List[complex]
RANDOM_VALUES_SYMBOLS_TYPE = List[ List[ List[float] ] ]
RANDOM_VALUES_CIR_TYPE = List[ List[ List[float] ] ]
NOISY_SYMBOL_SEQUENCE_TYPE = List[ List[complex] ]
SER_TYPE = Union[float, None]
BER_TYPE = Union[float, None]

#########################################################################
#                   Given Modulation Bit to Symbol Maps                 #
#########################################################################

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

#########################################################################
#                           Evaluation Function                         #
#########################################################################

def evaluate():
    """
    Your code used to evaluate your system should be written here.
             !!! NOTE: This function will not be marked !!!
    """


    return
   
#########################################################################
#                           Assisting Functions                         #
#########################################################################

def assist_bit_to_symbol(bit_sequence: BIT_SEQUENCE_TYPE, bit_to_symbol_map: BIT_TO_SYMBOL_MAP_TYPE) -> SYMBOL_SEQUENCE_TYPE:
    """
    Converts a sequence of bits to a sequence of symbols using the bit to symbol map.

    parameters:
        bit_sequence -> type <class 'list'> : List containing int items which represents the bits for example: [0, 1, 1, 1, 0, 0, 1, 1]   
        bit_to_symbol_map -> type <class 'list'> : A list containing lists. Each list entry contains a complex number and a bit sequence, representing the symbol, and the corresponding bit sequence that maps to it.
                             the maps that will be used are given as MAP_BPSK and MAP_4QAM
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



def assist_symbol_to_bit(symbol_sequence: SYMBOL_SEQUENCE_TYPE, bit_to_symbol_map: BIT_TO_SYMBOL_MAP_TYPE) -> BIT_SEQUENCE_TYPE:
    """
    Returns a sequence of bits that corresponds to the provided sequence of symbols containing noise using the bit to symbol map that respresent the modulation scheme and the euclidean distance

    parameters:
        symbol_sequence -> type <class 'list'> : List containing complex items (<class "complex">) which represents the symbols for example: [1+1j, 2+2j, -0.66-0.25j]
        bit_to_symbol_map -> type <class 'list'> : A list containing lists. Each list entry contains a complex number and a bit sequence, representing the symbol, and the corresponding bit sequence that maps to it.
                             the maps that will be used are given as MAP_BPSK and MAP_4QAM
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

def assist_split_symbols_into_blocks(symbol_sequence: SYMBOL_SEQUENCE_TYPE, block_size: int) -> SYMBOL_BLOCKS_TYPE:
    """
    Divides the given symbol sequence into blocks of length block_size, that the DFE and MLSE algorithm should be performed upon.

    parameters:
        symbol_sequence -> type <class 'list'> : List containing complex items (<class "complex">) which represents the symbols for example: [1+1j, 2+2j, -0.66-0.25j]
        block_size -> type <class 'int'> : An integer indicating the amount of symbols that should be within each block
        
    returns:
        symbol_blocks -> type <class 'list'> : List of lists. Each list entry should be a list representing a symbol sequence, which is a list containing containing complex items (<class "complex">) which represents the symbols for example: [1+1j, 2+2j, -0.66-0.25j]
          Example, blocks of size 3 :
            [
                [(-0.262+0.757j), ( 0.071+0.161j), (0.255-0.333j)],
                [( 0.247-0.471j), (-0.668-0.256j), (0.032-0.345j)],
                [(-0.256-0.217j), ( 0.577-0.327j), (0.347-0.359j)],
                [(-0.091+0.641j), ( 0.535+0.101j), (0.203+0.614j)],
            ]
    """

    symbol_blocks = [symbol_sequence[i:i + block_size] for i in range(0, len(symbol_sequence), block_size)]
    return symbol_blocks

def assist_combine_blocks_into_symbols(symbol_blocks: SYMBOL_BLOCKS_TYPE) -> SYMBOL_SEQUENCE_TYPE:
    """
    Combines the given blocks of symbol sequences into a single sequence of symbols.

    parameters:
        symbol_blocks -> type <class 'list'> : List of lists. Each list entry should be a list representing a symbol sequence, which is a list containing containing complex items (<class "complex">) which represents the symbols for example: [1+1j, 2+2j, -0.66-0.25j]
          Example, blocks of size 3 :
            [
                [(-0.262+0.757j), ( 0.071+0.161j), (0.255-0.333j)],
                [( 0.247-0.471j), (-0.668-0.256j), (0.032-0.345j)],
                [(-0.256-0.217j), ( 0.577-0.327j), (0.347-0.359j)],
                [(-0.091+0.641j), ( 0.535+0.101j), (0.203+0.614j)],
            ]

    returns:
        symbol_sequence -> type <class 'list'> : List containing complex items (<class "complex">) which represents the symbols for example: [1+1j, 2+2j, -0.66-0.25j]

    """

    symbol_sequence = [symbol for block in symbol_blocks for symbol in block]
    return symbol_sequence

#########################################################################
#                         DFE and MLSE Functions                        #
#########################################################################

def DFE_BPSK_BLOCK(symbol_sequence: SYMBOL_SEQUENCE_TYPE, impulse_response: CHANNEL_IMPULSE_RESPONSE_TYPE) -> SYMBOL_SEQUENCE_TYPE:
    """
    Performs the DFE algorithm on the given symbol sequence (which was modulated using the BPSK scheme) with the given impulse response, and returns the most probable transmitted symbol sequence.
    The impulse response length can be assumed to be 3, and the prepended symbols for t=-1 and t=-2 can be assumed to be [1, 1]
    Only the transmitted data bits must be returned, thus exluding the prepended symbols. Thus len(symbol_sequence) equals len(transmitted_sequence).

    parameters:
        symbol_sequence -> type <class 'list'> : List containing complex items (<class "complex">) which represents the symbols for example: [1+1j, 2+2j, -0.66-0.25j]
        impulse_response -> type <class 'list'> : List containing complex items (<class "complex">) which represents the impulse response coeficients for example [1+1j, 2+2j, -0.66-0.25j] represents [c0, c1, c2] 

    returns:
        transmitted_sequence -> type <class 'list'> : List containing complex items (<class "complex">) which represents the symbols for example: [1+1j, 2+2j, -0.66-0.25j]

    """

    transmitted_sequence = []
    past_symbols = [1, 1]  # Predefined past symbols
    for idx, received in enumerate(symbol_sequence):
 
        delta_pos = abs(received - (impulse_response[0] * 1 + impulse_response[1] * past_symbols[-1] + impulse_response[2] * past_symbols[-2])) ** 2
        delta_neg = abs(received - (impulse_response[0] * -1 + impulse_response[1] * past_symbols[-1] + impulse_response[2] * past_symbols[-2])) ** 2

        if delta_pos < delta_neg:
            transmitted_sequence.append(1)
            past_symbols.append(1)
        else:
            transmitted_sequence.append(-1)
            past_symbols.append(-1)

    return transmitted_sequence


def DFE_4QAM_BLOCK(symbol_sequence: SYMBOL_SEQUENCE_TYPE, impulse_response: CHANNEL_IMPULSE_RESPONSE_TYPE) -> SYMBOL_SEQUENCE_TYPE:
    """
    Performs the DFE algorithm on the given symbol sequence (which was modulated using the 4QAM scheme) with the given impulse response, and returns the most probable transmitted symbol sequence.
    The impulse response length can be assumed to be 3, and the prepended symbols for t=-1 and t=-2 can be assumed to be [(0.7071067811865475+0.7071067811865475j), (0.7071067811865475+0.7071067811865475j)]
    Only the transmitted data bits must be returned, thus exluding the prepended symbols. Thus len(symbol_sequence) equals len(transmitted_sequence).

    parameters:
        symbol_sequence -> type <class 'list'> : List containing complex items (<class "complex">) which represents the symbols for example: [1+1j, 2+2j, -0.66-0.25j]
        impulse_response -> type <class 'list'> : List containing complex items (<class "complex">) which represents the impulse response coeficients for example [1+1j, 2+2j, -0.66-0.25j] represents [c0, c1, c2] 

    returns:
        transmitted_sequence -> type <class 'list'> : List containing complex items (<class "complex">) which represents the symbols for example: [1+1j, 2+2j, -0.66-0.25j]

    """

    transmitted_sequence = []
    past_symbols = [(0.7071067811865475+0.7071067811865475j), (0.7071067811865475+0.7071067811865475j)]
    constellation_points = [
        (1+1j) / cm.sqrt(2), (-1+1j) / cm.sqrt(2),
        (1-1j) / cm.sqrt(2), (-1-1j) / cm.sqrt(2)
    ]

    for idx, received in enumerate(symbol_sequence):
        # Compute distance metric for each constellation point
        distances = [
            abs(received - (impulse_response[0] * point + impulse_response[1] * past_symbols[-1] + impulse_response[2] * past_symbols[-2])) ** 2 for point in constellation_points
        ]
        # Choose the point with the smallest error
        best_point = constellation_points[distances.index(min(distances))]
        transmitted_sequence.append(best_point)
        past_symbols.append(best_point)

    return transmitted_sequence

def MLSE_BPSK_BLOCK(symbol_sequence: SYMBOL_SEQUENCE_TYPE, impulse_response: CHANNEL_IMPULSE_RESPONSE_TYPE) -> SYMBOL_SEQUENCE_TYPE:
    """
    Performs the MLSE algorithm on the given symbol sequence (which was modulated using the BPSK scheme) with the given impulse response, and returns the most probable transmitted symbol sequence.
    The impulse response length can be assumed to be 3, and the prepended symbols for t=-1 and t=-2 can be assumed to be [1, 1]
    
    !!! NOTE: The appended symbols should be included in the given symbol sequence, thus if the block size is 200, then the length of the given symbol sequence should be 202. 
    
    Only the transmitted data bits must be returned, thus exluding the prepended symbols AND the appended symbols. Thus is the block size is 200 then len(transmitted_sequence) should be 200.

    parameters:
        symbol_sequence -> type <class 'list'> : List containing complex items (<class "complex">) which represents the symbols for example: [1+1j, 2+2j, -0.66-0.25j]
        impulse_response -> type <class 'list'> : List containing complex items (<class "complex">) which represents the impulse response coeficients for example [1+1j, 2+2j, -0.66-0.25j] represents [c0, c1, c2] 

    returns:
        transmitted_sequence -> type <class 'list'> : List containing complex items (<class "complex">) which represents the symbols for example: [1+1j, 2+2j, -0.66-0.25j]

    """
    
    trellis_length = len(symbol_sequence) + 1
    L = len(impulse_response)
    class Node:
        def __init__(self, t, state):
            self.t = t
            self.children = []
            self.parents = []
            self.delta = 0
            self.nodePath = []
            self.state = state

    

    def getSymbol(state):
        state = state // 2
        symbols = [1 + 0j, -1 + 0j]
        return symbols[state]

    def getNextStates(state):
        states = [
            [0, 2],
            [0, 2],
            [1, 3],
            [1, 3]
        ]
        return states[state]
    

    def getNodes(t):
        if t == 0 or t == trellis_length - 1:
            return [Node(t, 0)]
        if t == 1:
            return [Node(t, 0), Node(t, 2)]
        if t == trellis_length - 2:
            return [Node(t, 0), Node(t, 1)]
        return [Node(t, s) for s in range(4)]

    trellis = []

    def calculate_deltas(node, path):
        if len(node.parents):
            path = path[:]
            path.append(node)
            while len(path) < L:
                path.insert(0, Node(path[0].t - 1, 0))

            delta = symbol_sequence[node.t - 1]
            for n in range(L):
                delta -= getSymbol(path[- 1 - n].state) * impulse_response[n]
            delta = abs(delta) ** 2
            delta += path[-2].delta

            if node.delta:
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

    def createTrellis():
        for t in range(trellis_length):
            trellis.append(getNodes(t))

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

    def get_path(node):
        path = []
        for path_node in node.nodePath:
            path.append(getSymbol(path_node.state))

        for i in range(L - 1):
            path.pop(0)
            path.pop()

        return path

    root, leaf = createTrellis()
    calculate_deltas(root, root.nodePath)
    transmitted_sequence = get_path(leaf)

    return transmitted_sequence


def MLSE_4QAM_BLOCK(symbol_sequence: SYMBOL_SEQUENCE_TYPE, impulse_response: CHANNEL_IMPULSE_RESPONSE_TYPE) -> SYMBOL_SEQUENCE_TYPE:
    """
    Performs the MLSE algorithm on the given symbol sequence (which was modulated using the 4QAM scheme) with the given impulse response, and returns the most probable transmitted symbol sequence.
    The impulse response length can be assumed to be 3, and the prepended symbols for t=-1 and t=-2 can be assumed to be [(0.7071067811865475+0.7071067811865475j), (0.7071067811865475+0.7071067811865475j)]
    
    !!! NOTE: The appended symbols should be included in the given symbol sequence, thus if the block size is 200, then the length of the given symbol sequence should be 202. 
    
    Only the transmitted data bits must be returned, thus exluding the prepended symbols AND the appended symbols. Thus is the block size is 200 then len(transmitted_sequence) should be 200.

    parameters:
        symbol_sequence -> type <class 'list'> : List containing complex items (<class "complex">) which represents the symbols for example: [1+1j, 2+2j, -0.66-0.25j]
        impulse_response -> type <class 'list'> : List containing complex items (<class "complex">) which represents the impulse response coeficients for example [1+1j, 2+2j, -0.66-0.25j] represents [c0, c1, c2] 

    returns:
        transmitted_sequence -> type <class 'list'> : List containing complex items (<class "complex">) which represents the symbols for example: [1+1j, 2+2j, -0.66-0.25j]

    """
    trellis_length = len(symbol_sequence) + 1
    L = len(impulse_response)
    class Node:
        def __init__(self, t, state):
            self.t = t
            self.children = []
            self.parents = []
            self.delta = 0
            self.nodePath = []
            self.state = state

    def getNextStates(state):
        states = [
            *[[0, 4, 8, 12]] * 4,
            *[[1, 5, 9, 13]] * 4,
            *[[2, 6, 10, 14]] * 4,
            *[[3, 7, 11, 15]] * 4,
        ]
        return states[state]

   

    

    def getNodes(t):
        if t == 0 or t == trellis_length - 1:
            return [Node(t, 0)]
        if t == 1:
            return [Node(t, 0), Node(t, 4), Node(t, 8), Node(t, 12)]
        if t == trellis_length - 2:
            return [Node(t, 0), Node(t, 1), Node(t, 2), Node(t, 3)]
        return [Node(t, s) for s in range(16)]

    def getSymbol(state):
        state = state // 4
        symbols = [(1 + 1j) / cm.sqrt(2), (-1 + 1j) / cm.sqrt(2), (-1 - 1j) / cm.sqrt(2), (1 - 1j) / cm.sqrt(2)]
        return symbols[state]

    def calculate_deltas(node, path):
            
        if len(node.parents):
            # ensure node path is full
            path = path[:]
            path.append(node)
            while len(path) < L:
                path.insert(0, Node(path[0].t - 1, 0))

            # calculate node delta for node path
            delta = symbol_sequence[node.t - 1]
            for n in range(L):
                delta -= getSymbol(path[- 1 - n].state) * impulse_response[n]
            delta = m.pow(abs(delta), 2)
            # add previous node delta to current delta
            delta += path[-2].delta

            # compare new and current delta values
            if node.delta:
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
            path.append(getSymbol(path_node.state))

        for i in range(L - 1):
            path.pop(0)
            path.pop()

        return path

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

   
    



    root, leaf = createTrellis()
    calculate_deltas(root, root.nodePath)
    transmitted_sequence = get_path(leaf)


    return transmitted_sequence


#########################################################################
#                         SER and BER Functions                         #
#########################################################################

# BPSK

def SER_BER_BPSK_DFE_STATIC(bit_sequence: BIT_SEQUENCE_TYPE, block_size: int, random_values_for_symbols: RANDOM_VALUES_SYMBOLS_TYPE, snr: float) -> Tuple[SYMBOL_BLOCKS_TYPE, SER_TYPE, BER_TYPE]:
    """
    This function uses the given bit sequence, block size, SNR value and random values to perform a simulation that:
        - converts the bit sequence into a symbol sequence using BPSK
        - splits the symbol sequence into blocks with the given block size
        - add noise to the symbol sequence in each block using the equation in the practical guide and the static impulse response and SNR
        - save the noisy symbol sequence from each block
        - perform the DFE algorithm on each noisy symbol sequence in the blocks to get the corresponding transmitted sequences
        - combine the blocks of transmitted sequences into a single symbol sequence
        - calculate the SER and BER by comparing the given sequences and transmitted sequences
        - return the blocks of noisy symbols, the SER value scale to log10, and the BER value scale to log10

    parameters:
        bit_sequence -> type <class 'list'> : List containing int items which represents the bits for example: [0, 1, 1, 1, 0, 0, 1, 1]  
        block_size -> type <class 'int'> : An integer indicating the amount of symbols that should be within each block
        random_values_for_symbols -> type <class 'list'> : A list containing lists containing lists. Thus the variable has three index values random_values_for_symbols[0][0][0] and returns a float.
            These random values are used to calculate the added noise according to equation 1 given in the practical guide.
            The first index indicates the corresponding block. The second index indicate the corresponding symbol in that block. The third index indicate the real (0) or imaginary (1) component for which the float value should be used.
            !!! NOTE: The imaginary component in this case will be 0 since for BPSK no imaginary noise will be added and the other equation shown in the pre-practical slides should be used.
            Thus -> random_values_for_symbols[5][84][0] -> is the float that should be used for the real component kappa value to calculate the noise for the 85th symbol in the 6th block (note: the first block is index 0).
        snr -> type <class 'float'> : A float value giving the SNR value that should be used for calculating the sigma value used in the noise equation.

    returns:
        noisy_symbol_blocks -> type <class 'list'> : List containing lists. Each list entry contains a sequence of <class "complex"> which represents the symbols after noise was added for each block
          Example, blocks of size 3 :
            [
                [(-0.262+0.757j), ( 0.071+0.161j), (0.255-0.333j)],
                [(-0.256-0.217j), ( 0.577-0.327j), (0.347-0.359j)],
                [( 0.247-0.471j), (-0.668-0.256j), (0.032-0.345j)],
                [(-0.091+0.641j), ( 0.535+0.101j), (0.203+0.614j)],
            ]
        SER -> type <class 'float'> or <class 'NoneType'> : Returns the SER value (not scaled) or returns the NoneType (return None) if the amount of errors is zero. 
        BER -> type <class 'float'> or <class 'NoneType'> : Returns the BER value (not scaled) or returns the NoneType (return None) if the amount of errors is zero. 
            
    """

   
    channel_response = [0.29 + 0.98j, 0.73 - 0.24j, 0.21 + 0.91j]
    
    symbols = np.array([1 if bit == 1 else -1 for bit in bit_sequence])

    energy_per_bit_to_noise = 10 ** (snr / 10)
    noise_variance = np.sqrt(1 / (2 * energy_per_bit_to_noise))

    total_blocks = len(symbols) // block_size
    noisy_symbols = []

    for block_index in range(total_blocks):
        noise_affected_block = []
        for time in range(block_size):
            current_symbol = symbols[block_index * block_size + time]
            prev_symbol_1 = symbols[block_index * block_size + time - 1] if time > 0 else 0
            prev_symbol_2 = symbols[block_index * block_size + time - 2] if time > 1 else 0

            noise_real = random_values_for_symbols[block_index][time][0] * noise_variance
            noise_imaginary = random_values_for_symbols[block_index][time][1] * noise_variance
            
            combined_noise = noise_real + 1j * noise_imaginary
            received_signal = current_symbol * channel_response[0] + prev_symbol_1 * channel_response[1] + prev_symbol_2 * channel_response[2] + combined_noise
            
            noise_affected_block.append(received_signal)
        
        noisy_symbols.append(noise_affected_block)

    transmitted_symbols = []
    for noisy_block in noisy_symbols:
        transmitted_block = []
        for received in noisy_block:
            transmitted_block.append(1 if np.real(received) >= 0 else -1)
        transmitted_symbols.extend(transmitted_block)

    symbol_error_count = np.sum(np.array(transmitted_symbols) != symbols)
    bit_error_count = symbol_error_count  

    ser_result = symbol_error_count / len(symbols)
    ber_result = bit_error_count / len(symbols)

    return noisy_symbols, ser_result, ber_result

def SER_BER_BPSK_DFE_DYNAMIC(bit_sequence: BIT_SEQUENCE_TYPE, block_size: int, random_values_for_symbols: RANDOM_VALUES_SYMBOLS_TYPE, random_values_for_CIR: RANDOM_VALUES_CIR_TYPE, snr: float) -> Tuple[SYMBOL_BLOCKS_TYPE, SER_TYPE, BER_TYPE]:
    """
    This function uses the given bit sequence, block size, SNR value and random values to perform a simulation that:
        - converts the bit sequence into a symbol sequence using BPSK
        - splits the symbol sequence into blocks with the given block size
        - calculate a channel impulse response for each block using the dynamic impulse response equation in the practical guide
        - add noise to the symbol sequence in each block using the equation in the practical guide and the dynamic impulse response and SNR
        - save the noisy symbol sequence from each block
        - perform the DFE algorithm on each noisy symbol sequence in the blocks to get the corresponding transmitted sequences
        - combine the blocks of transmitted sequences into a single symbol sequence
        - calculate the SER and BER by comparing the given sequences and transmitted sequences
        - return the blocks of noisy symbols, the SER value scale to log10, and the BER value scale to log10

    parameters:
        bit_sequence -> type <class 'list'> : List containing int items which represents the bits for example: [0, 1, 1, 1, 0, 0, 1, 1]  
        block_size -> type <class 'int'> : An integer indicating the amount of symbols that should be within each block
        random_values_for_symbols -> type <class 'list'> : A list containing lists containing lists. Thus the variable has three index values random_values_for_symbols[0][0][0] and returns a float.
            These random values are used to calculate the added noise according to equation 1 given in the practical guide.
            The first index indicates the corresponding block. The second index indicate the corresponding symbol in that block. The third index indicate the real (0) or imaginary (1) component for which the float value should be used.
            !!! NOTE: The imaginary component in this case will be 0 since for BPSK no imaginary noise will be added and the other equation shown in the pre-practical slides should be used.
            Thus -> random_values_for_symbols[5][84][0] -> is the float that should be used for the real component kappa value to calculate the noise for the 85th symbol in the 6th block (note: the first block is index 0).
        random_values_for_CIR -> type <class 'list'> : A list containing lists containing lists. Thus the variable has three index values random_values_for_CIR[0][0][0] and returns a float.
            These random values are used to calculate the dynamic impulse response using equation 4 in the practical guide
            The first index indicates the corresponding block. The second index indicate the corresponding coeficient (c0 - [0], c1 = [1], c2 - [2]). The third index indicates the real (0) or imaginary (1) component for which the float value should be used.
            Thus -> random_values_for_CIR[7][2][1] -> is the float that should be used for the imaginary component kappa value to calculate the c2 coeficient for the 8th block (note: the first block is index 0).
        snr -> type <class 'float'> : A float value giving the SNR value that should be used for calculating the sigma value used in the noise equation.

    returns:
        noisy_symbol_blocks -> type <class 'list'> : List containing lists. Each list entry contains a sequence of <class "complex"> which represents the symbols after noise was added for each block
          Example, blocks of size 3 :
            [
                [(-0.256-0.217j), ( 0.577-0.327j), (0.347-0.359j)],
                [(-0.262+0.757j), ( 0.071+0.161j), (0.255-0.333j)],
                [( 0.247-0.471j), (-0.668-0.256j), (0.032-0.345j)],
                [(-0.091+0.641j), ( 0.535+0.101j), (0.203+0.614j)],
            ]
        SER -> type <class 'float'> or <class 'NoneType'> : Returns the SER value (not scaled) or returns the NoneType (return None) if the amount of errors is zero. 
        BER -> type <class 'float'> or <class 'NoneType'> : Returns the BER value (not scaled) or returns the NoneType (return None) if the amount of errors is zero.  
            
    """

    noisy_symbol_data = []
    ser_value = 0
    ber_value = 0

    modulated_symbols = assist_bit_to_symbol(bit_sequence, MAP_BPSK)
    symbol_blocks = assist_split_symbols_into_blocks(modulated_symbols, block_size)

    bit_energy = m.log2(len(MAP_BPSK))
    noise_factor = 1 / (m.sqrt(pow(10, (0.1 * snr)) * bit_energy))

    for idx, block in enumerate(symbol_blocks):
        noisy_block = []
        channel_coeff_0 = (complex(random_values_for_CIR[idx][0][0], random_values_for_CIR[idx][0][1])) / m.sqrt(6)
        channel_coeff_1 = (complex(random_values_for_CIR[idx][1][0], random_values_for_CIR[idx][1][1])) / m.sqrt(6)
        channel_coeff_2 = (complex(random_values_for_CIR[idx][2][0], random_values_for_CIR[idx][2][1])) / m.sqrt(6)
        cir = [channel_coeff_0, channel_coeff_1, channel_coeff_2]
        
        padded_block = [complex(1, 0), complex(1, 0)] + block
        for z in range(len(block)):
            noisy_signal = padded_block[z + 2] * cir[0] + padded_block[z + 1] * cir[1] + padded_block[z] * cir[2] + (noise_factor * (random_values_for_symbols[idx][z][0] + 0))
            noisy_block.append(noisy_signal)

        noisy_symbol_data.append(noisy_block)

    processed_blocks = []
    for idx, block in enumerate(noisy_symbol_data):
        cir = [
            (complex(random_values_for_CIR[idx][0][0], random_values_for_CIR[idx][0][1])) / m.sqrt(6),
            (complex(random_values_for_CIR[idx][1][0], random_values_for_CIR[idx][1][1])) / m.sqrt(6),
            (complex(random_values_for_CIR[idx][2][0], random_values_for_CIR[idx][2][1])) / m.sqrt(6)
        ]
        processed_blocks.append(DFE_BPSK_BLOCK(block, cir))

    demodulated_symbols = assist_combine_blocks_into_symbols(processed_blocks)

    symbol_error_ratio = sum([1 for i in range(len(modulated_symbols)) if modulated_symbols[i] != demodulated_symbols[i]]) / len(modulated_symbols)
    ser_value = None if symbol_error_ratio == 0 else symbol_error_ratio

    demodulated_bits = assist_symbol_to_bit(demodulated_symbols, MAP_BPSK)

    bit_error_ratio = sum([1 for i in range(len(bit_sequence)) if bit_sequence[i] != demodulated_bits[i]]) / len(bit_sequence)
    ber_value = None if bit_error_ratio == 0 else bit_error_ratio

    return noisy_symbol_data, ser_value, ber_value

def SER_BER_BPSK_MLSE_STATIC(bit_sequence: BIT_SEQUENCE_TYPE, block_size: int, random_values_for_symbols: RANDOM_VALUES_SYMBOLS_TYPE, snr: float) -> Tuple[SYMBOL_BLOCKS_TYPE, SER_TYPE, BER_TYPE]:
    """
    This function uses the given bit sequence, block size, SNR value and random values to perform a simulation that:
        - converts the bit sequence into a symbol sequence using BPSK
        - splits the symbol sequence into blocks with the given block size
        - add the appended symbols to each sequence in each block. For BPSK the appended symbols are [1, 1]
        - add noise to the symbol sequence including the appended symbols in each block using the equation in the practical guide and the static impulse response and SNR.
        - save the noisy symbol sequence from each block which should all be block size + 2 long, thus including the symbols that was appended.
        - perform the MLSE algorithm on each noisy symbol sequence in the blocks to get the corresponding transmitted sequences excluding the appended symbols.
        - combine the blocks of transmitted sequences into a single symbol sequence
        - calculate the SER and BER by comparing the given sequences and transmitted sequences
        - return the blocks of noisy symbols, the SER value scale to log10, and the BER value scale to log10

    parameters:
        bit_sequence -> type <class 'list'> : List containing int items which represents the bits for example: [0, 1, 1, 1, 0, 0, 1, 1]  
        block_size -> type <class 'int'> : An integer indicating the amount of symbols that should be within each block
        random_values_for_symbols -> type <class 'list'> : A list containing lists containing lists. Thus the variable has three index values random_values_for_symbols[0][0][0] and returns a float.
            These random values are used to calculate the added noise according to equation 1 given in the practical guide.
            The first index indicates the corresponding block. The second index indicate the corresponding symbol in that block. The third index indicate the real (0) or imaginary (1) component for which the float value should be used.
            !!! NOTE: The imaginary component in this case will be 0 since for BPSK no imaginary noise will be added and the other equation shown in the pre-practical slides should be used.
            Thus -> random_values_for_symbols[5][84][0] -> is the float that should be used for the real component kappa value to calculate the noise for the 85th symbol in the 6th block (note: the first block is index 0).
        snr -> type <class 'float'> : A float value giving the SNR value that should be used for calculating the sigma value used in the noise equation.

    returns:
        noisy_symbol_blocks -> type <class 'list'> : List containing lists. Each list entry contains a sequence of <class "complex"> which represents the symbols after noise was added for each block
          Example, blocks of size 3 :
            [
                [(-0.262+0.757j), ( 0.071+0.161j), (0.255-0.333j)],
                [(-0.256-0.217j), ( 0.577-0.327j), (0.347-0.359j)],
                [( 0.247-0.471j), (-0.668-0.256j), (0.032-0.345j)],
                [(-0.091+0.641j), ( 0.535+0.101j), (0.203+0.614j)],
            ]
        SER -> type <class 'float'> or <class 'NoneType'> : Returns the SER value (not scaled) or returns the NoneType (return None) if the amount of errors is zero. 
        BER -> type <class 'float'> or <class 'NoneType'> : Returns the BER value (not scaled) or returns the NoneType (return None) if the amount of errors is zero. 
            
    """

    noisy_symbol_blocks = []
    SER = 0
    BER = 0

    symbols = assist_bit_to_symbol(bit_sequence, MAP_BPSK)
    Blocks = assist_split_symbols_into_blocks(symbols, block_size)

    CIR = [complex(0.29, 0.98), complex(0.73, -0.24), complex(0.21, 0.91)]

    fbit = m.log2(len(MAP_BPSK))
    sigma = 1 / (m.sqrt(pow(10, (0.1 * snr)) * fbit))

    for i in range(len(Blocks)):
        temp = []
        temp = Blocks[i]
        noisy = []
        for z in range(len(temp)):
            temp = [complex(1, 0), complex(1, 0)]
            temp.extend(Blocks[i])
            noisy.append(temp[z + 2] * CIR[0] + temp[z + 1] * CIR[1] + temp[z] * CIR[2] + (sigma * (random_values_for_symbols[i][z][0] + 0)))
        noisy_symbol_blocks.append(noisy)

    newBlocks = []
    Tosend = noisy_symbol_blocks.copy()
    for n in Tosend:
        n.append(1)
        n.append(1)
        newBlocks.append(MLSE_BPSK_BLOCK(n, CIR))

    correctedSymbols = assist_combine_blocks_into_symbols(newBlocks)

    for k in range(len(symbols)):
        if symbols[k] != correctedSymbols[k]:
            SER += 1
    SER /= len(symbols)
    if SER == 0:
        SER = None

    noisyBits = assist_symbol_to_bit(correctedSymbols, MAP_BPSK)

    for l in range(len(bit_sequence)):
        if bit_sequence[l] != noisyBits[l]:
            BER += 1
    BER /= len(bit_sequence)
    if BER == 0:
        BER = None

    return noisy_symbol_blocks, SER, BER

def SER_BER_BPSK_MLSE_DYNAMIC(bit_sequence: BIT_SEQUENCE_TYPE, block_size: int, random_values_for_symbols: RANDOM_VALUES_SYMBOLS_TYPE, random_values_for_CIR: RANDOM_VALUES_CIR_TYPE, snr: float) -> Tuple[SYMBOL_BLOCKS_TYPE, SER_TYPE, BER_TYPE]:
    """
    This function uses the given bit sequence, block size, SNR value and random values to perform a simulation that:
        - converts the bit sequence into a symbol sequence using BPSK
        - splits the symbol sequence into blocks with the given block size
        - add the appended symbols to each sequence in each block. For BPSK the appended symbols are [1, 1]
        - calculate a channel impulse response for each block using the dynamic impulse response equation in the practical guide
        - add noise to the symbol sequence including the appended symbols in each block using the equation in the practical guide and the static impulse response and SNR.
        - save the noisy symbol sequence from each block which should all be block size + 2 long, thus including the symbols that was appended.
        - perform the MLSE algorithm on each noisy symbol sequence in the blocks to get the corresponding transmitted sequences excluding the appended symbols.
        - combine the blocks of transmitted sequences into a single symbol sequence
        - calculate the SER and BER by comparing the given sequences and transmitted sequences
        - return the blocks of noisy symbols, the SER value scale to log10, and the BER value scale to log10

    parameters:
        bit_sequence -> type <class 'list'> : List containing int items which represents the bits for example: [0, 1, 1, 1, 0, 0, 1, 1]  
        block_size -> type <class 'int'> : An integer indicating the amount of symbols that should be within each block
        random_values_for_symbols -> type <class 'list'> : A list containing lists containing lists. Thus the variable has three index values random_values_for_symbols[0][0][0] and returns a float.
            These random values are used to calculate the added noise according to equation 1 given in the practical guide.
            The first index indicates the corresponding block. The second index indicate the corresponding symbol in that block. The third index indicate the real (0) or imaginary (1) component for which the float value should be used.
            !!! NOTE: The imaginary component in this case will be 0 since for BPSK no imaginary noise will be added and the other equation shown in the pre-practical slides should be used.
            Thus -> random_values_for_symbols[5][84][0] -> is the float that should be used for the real component kappa value to calculate the noise for the 85th symbol in the 6th block (note: the first block is index 0).
        random_values_for_CIR -> type <class 'list'> : A list containing lists containing lists. Thus the variable has three index values random_values_for_CIR[0][0][0] and returns a float.
            These random values are used to calculate the dynamic impulse response using equation 4 in the practical guide
            The first index indicates the corresponding block. The second index indicate the corresponding coeficient (c0 - [0], c1 = [1], c2 - [2]). The third index indicates the real (0) or imaginary (1) component for which the float value should be used.
            Thus -> random_values_for_CIR[7][2][1] -> is the float that should be used for the imaginary component kappa value to calculate the c2 coeficient for the 8th block (note: the first block is index 0).
        snr -> type <class 'float'> : A float value giving the SNR value that should be used for calculating the sigma value used in the noise equation.

    returns:
        noisy_symbol_blocks -> type <class 'list'> : List containing lists. Each list entry contains a sequence of <class "complex"> which represents the symbols after noise was added for each block
          Example, blocks of size 3 :
            [
                [(-0.256-0.217j), ( 0.577-0.327j), (0.347-0.359j)],
                [(-0.262+0.757j), ( 0.071+0.161j), (0.255-0.333j)],
                [( 0.247-0.471j), (-0.668-0.256j), (0.032-0.345j)],
                [(-0.091+0.641j), ( 0.535+0.101j), (0.203+0.614j)],
            ]
        SER -> type <class 'float'> or <class 'NoneType'> : Returns the SER value (not scaled) or returns the NoneType (return None) if the amount of errors is zero. 
        BER -> type <class 'float'> or <class 'NoneType'> : Returns the BER value (not scaled) or returns the NoneType (return None) if the amount of errors is zero. 
            
    """

    noisy_symbol_blocks = []
    SER = 0
    BER = 0

    symbols = assist_bit_to_symbol(bit_sequence, MAP_BPSK)
    Blocks = assist_split_symbols_into_blocks(symbols, block_size)

    fbit = m.log2(len(MAP_BPSK))
    sigma = 1 / (m.sqrt(pow(10, (0.1 * snr)) * fbit))

    for i in range(len(Blocks)):
        temp = []
        temp = Blocks[i]
        noisy = []

        c0 = (complex(random_values_for_CIR[i][0][0], random_values_for_CIR[i][0][1])) / m.sqrt(6)
        c1 = (complex(random_values_for_CIR[i][1][0], random_values_for_CIR[i][1][1])) / m.sqrt(6)
        c2 = (complex(random_values_for_CIR[i][2][0], random_values_for_CIR[i][2][1])) / m.sqrt(6)
        CIR = [c0, c1, c2]

        for z in range(len(temp)):
            temp = [complex(1, 0), complex(1, 0)]
            temp.extend(Blocks[i])
            noisy.append(temp[z + 2] * CIR[0] + temp[z + 1] * CIR[1] + temp[z] * CIR[2] + (sigma * (random_values_for_symbols[i][z][0] + 0)))

        noisy_symbol_blocks.append(noisy)

    newBlocks = []
    Tosend = noisy_symbol_blocks.copy()
    itest = 0
    for n in Tosend:
        c0 = (complex(random_values_for_CIR[itest][0][0], random_values_for_CIR[itest][0][1])) / m.sqrt(6)
        c1 = (complex(random_values_for_CIR[itest][1][0], random_values_for_CIR[itest][1][1])) / m.sqrt(6)
        c2 = (complex(random_values_for_CIR[itest][2][0], random_values_for_CIR[itest][2][1])) / m.sqrt(6)
        n.append(complex(1, 0))
        n.append(complex(1, 0))
        itest += 1
        CIR = [c0, c1, c2]
        newBlocks.append(MLSE_BPSK_BLOCK(n, CIR))

    correctedSymbols = assist_combine_blocks_into_symbols(newBlocks)

    for k in range(len(symbols)):
        if symbols[k] != correctedSymbols[k]:
            SER += 1
    SER /= len(symbols)
    if SER == 0:
        SER = None

    noisyBits = assist_symbol_to_bit(correctedSymbols, MAP_BPSK)

    for l in range(len(bit_sequence)):
        if bit_sequence[l] != noisyBits[l]:
            BER += 1
    BER /= len(bit_sequence)
    if BER == 0:
        BER = None

    return noisy_symbol_blocks, SER, BER

# 4QAM

def SER_BER_4QAM_DFE_STATIC(bit_sequence: BIT_SEQUENCE_TYPE, block_size: int, random_values_for_symbols: RANDOM_VALUES_SYMBOLS_TYPE, snr: float) -> Tuple[SYMBOL_BLOCKS_TYPE, SER_TYPE, BER_TYPE]:
    """
    This function uses the given bit sequence, block size, SNR value and random values to perform a simulation that:
        - converts the bit sequence into a symbol sequence using 4QAM
        - splits the symbol sequence into blocks with the given block size
        - add noise to the symbol sequence in each block using the equation in the practical guide and the static impulse response and SNR
        - save the noisy symbol sequence from each block
        - perform the DFE algorithm on each noisy symbol sequence in the blocks to get the corresponding transmitted sequences
        - combine the blocks of transmitted sequences into a single symbol sequence
        - calculate the SER and BER by comparing the given sequences and transmitted sequences
        - return the blocks of noisy symbols, the SER value scale to log10, and the BER value scale to log10

    parameters:
        bit_sequence -> type <class 'list'> : List containing int items which represents the bits for example: [0, 1, 1, 1, 0, 0, 1, 1]  
        block_size -> type <class 'int'> : An integer indicating the amount of symbols that should be within each block
        random_values_for_symbols -> type <class 'list'> : A list containing lists containing lists. Thus the variable has three index values random_values_for_symbols[0][0][0] and returns a float.
            These random values are used to calculate the added noise according to equation 1 given in the practical guide.
            The first index indicates the corresponding block. The second index indicate the corresponding symbol in that block. The third index indicate the real (0) or imaginary (1) component for which the float value should be used.
            Thus -> random_values_for_symbols[5][84][0] -> is the float that should be used for the real component kappa value to calculate the noise for the 85th symbol in the 6th block (note: the first block is index 0).
        snr -> type <class 'float'> : A float value giving the SNR value that should be used for calculating the sigma value used in the noise equation.

    returns:
        noisy_symbol_blocks -> type <class 'list'> : List containing lists. Each list entry contains a sequence of <class "complex"> which represents the symbols after noise was added for each block
          Example, blocks of size 3 :
            [
                [(-0.256-0.217j), ( 0.577-0.327j), (0.347-0.359j)],
                [(-0.262+0.757j), ( 0.071+0.161j), (0.255-0.333j)],
                [( 0.247-0.471j), (-0.668-0.256j), (0.032-0.345j)],
                [(-0.091+0.641j), ( 0.535+0.101j), (0.203+0.614j)],
            ]
        SER -> type <class 'float'> or <class 'NoneType'> : Returns the SER value (not scaled) or returns the NoneType (return None) if the amount of errors is zero. 
        BER -> type <class 'float'> or <class 'NoneType'> : Returns the BER value (not scaled) or returns the NoneType (return None) if the amount of errors is zero. 
            
    """
    noisy_symbol_blocks = []
    SER = 0
    BER = 0

    symbols = assist_bit_to_symbol(bit_sequence, MAP_4QAM)
    blocks = assist_split_symbols_into_blocks(symbols, block_size)

    channel_response = [complex(0.29, 0.98), complex(0.73, -0.24), complex(0.21, 0.91)]

    fbit = m.log2(len(MAP_4QAM))
    sigma = 1 / (m.sqrt(pow(10, (0.1 * snr)) * fbit))

    for i, block in enumerate(blocks):
        noisy = []
        temp = [complex(0.7071067811865475, 0.7071067811865475), complex(0.7071067811865475, 0.7071067811865475)]
        temp.extend(block)
        for z, symbol in enumerate(block):
            noise = (sigma * (complex(random_values_for_symbols[i][z][0], random_values_for_symbols[i][z][1]) / m.sqrt(2)))
            noisy.append(temp[z + 2] * channel_response[0] + temp[z + 1] * channel_response[1] + temp[z] * channel_response[2] + noise)
        noisy_symbol_blocks.append(noisy)

    dfe_blocks = [DFE_4QAM_BLOCK(noisy_block, channel_response) for noisy_block in noisy_symbol_blocks]

    corrected_symbols = assist_combine_blocks_into_symbols(dfe_blocks)

    SER = sum(1 for s, cs in zip(symbols, corrected_symbols) if s != cs) / len(symbols)
    SER = None if SER == 0 else SER

    noisy_bits = assist_symbol_to_bit(corrected_symbols, MAP_4QAM)
    BER = sum(1 for b, nb in zip(bit_sequence, noisy_bits) if b != nb) / len(bit_sequence)
    BER = None if BER == 0 else BER

    return noisy_symbol_blocks, SER, BER

def SER_BER_4QAM_DFE_DYNAMIC(bit_sequence: BIT_SEQUENCE_TYPE, block_size: int, random_values_for_symbols: RANDOM_VALUES_SYMBOLS_TYPE, random_values_for_CIR: RANDOM_VALUES_CIR_TYPE, snr: float) -> Tuple[SYMBOL_BLOCKS_TYPE, SER_TYPE, BER_TYPE]:
    """
    This function uses the given bit sequence, block size, SNR value and random values to perform a simulation that:
        - converts the bit sequence into a symbol sequence using 4QAM
        - splits the symbol sequence into blocks with the given block size
        - calculate a channel impulse response for each block using the dynamic impulse response equation in the practical guide
        - add noise to the symbol sequence in each block using the equation in the practical guide and the dynamic impulse response and SNR
        - save the noisy symbol sequence from each block
        - perform the DFE algorithm on each noisy symbol sequence in the blocks to get the corresponding transmitted sequences
        - combine the blocks of transmitted sequences into a single symbol sequence
        - calculate the SER and BER by comparing the given sequences and transmitted sequences
        - return the blocks of noisy symbols, the SER value scale to log10, and the BER value scale to log10

    parameters:
        bit_sequence -> type <class 'list'> : List containing int items which represents the bits for example: [0, 1, 1, 1, 0, 0, 1, 1]  
        block_size -> type <class 'int'> : An integer indicating the amount of symbols that should be within each block
        random_values_for_symbols -> type <class 'list'> : A list containing lists containing lists. Thus the variable has three index values random_values_for_symbols[0][0][0] and returns a float.
            These random values are used to calculate the added noise according to equation 1 given in the practical guide.
            The first index indicates the corresponding block. The second index indicate the corresponding symbol in that block. The third index indicate the real (0) or imaginary (1) component for which the float value should be used.
            Thus -> random_values_for_symbols[5][84][0] -> is the float that should be used for the real component kappa value to calculate the noise for the 85th symbol in the 6th block (note: the first block is index 0).
        random_values_for_CIR -> type <class 'list'> : A list containing lists containing lists. Thus the variable has three index values random_values_for_CIR[0][0][0] and returns a float.
            These random values are used to calculate the dynamic impulse response using equation 4 in the practical guide
            The first index indicates the corresponding block. The second index indicate the corresponding coeficient (c0 - [0], c1 = [1], c2 - [2]). The third index indicates the real (0) or imaginary (1) component for which the float value should be used.
            Thus -> random_values_for_CIR[7][2][1] -> is the float that should be used for the imaginary component kappa value to calculate the c2 coeficient for the 8th block (note: the first block is index 0).
        snr -> type <class 'float'> : A float value giving the SNR value that should be used for calculating the sigma value used in the noise equation.

    returns:
        noisy_symbol_blocks -> type <class 'list'> : List containing lists. Each list entry contains a sequence of <class "complex"> which represents the symbols after noise was added for each block
          Example, blocks of size 3 :
            [
                [(-0.262+0.757j), ( 0.071+0.161j), (0.255-0.333j)],
                [(-0.256-0.217j), ( 0.577-0.327j), (0.347-0.359j)],
                [( 0.247-0.471j), (-0.668-0.256j), (0.032-0.345j)],
                [(-0.091+0.641j), ( 0.535+0.101j), (0.203+0.614j)],
            ]
        SER -> type <class 'float'> or <class 'NoneType'> : Returns the SER value (not scaled) or returns the NoneType (return None) if the amount of errors is zero. 
        BER -> type <class 'float'> or <class 'NoneType'> : Returns the BER value (not scaled) or returns the NoneType (return None) if the amount of errors is zero. 
            
    """
    noisy_symbol_blocks = []
    SER = 0
    BER = 0

    symbols = assist_bit_to_symbol(bit_sequence, MAP_4QAM)
    blocks = assist_split_symbols_into_blocks(symbols, block_size)

    fbit = m.log2(len(MAP_BPSK))
    sigma = 1 / (m.sqrt(pow(10, (0.1 * snr)) * fbit))

    for i, block in enumerate(blocks):
        noisy = []
        temp = [complex(1, 0), complex(1, 0)]
        temp.extend(block)
        cir = [complex(random_values_for_CIR[i][j][0], random_values_for_CIR[i][j][1]) / m.sqrt(6) for j in range(3)]
        for z, symbol in enumerate(block):
            noise = (sigma * (complex(random_values_for_symbols[i][z][0], random_values_for_symbols[i][z][1]) / m.sqrt(2)))
            noisy.append(temp[z + 2] * cir[0] + temp[z + 1] * cir[1] + temp[z] * cir[2] + noise)
        noisy_symbol_blocks.append(noisy)

    dfe_blocks = [DFE_4QAM_BLOCK(noisy_block, [complex(random_values_for_CIR[i][j][0], random_values_for_CIR[i][j][1]) / m.sqrt(6) for j in range(3)]) for i, noisy_block in enumerate(noisy_symbol_blocks)]

    corrected_symbols = assist_combine_blocks_into_symbols(dfe_blocks)

    SER = sum(1 for s, cs in zip(symbols, corrected_symbols) if s != cs) / len(symbols)
    SER = None if SER == 0 else SER

    noisy_bits = assist_symbol_to_bit(corrected_symbols, MAP_4QAM)
    BER = sum(1 for b, nb in zip(bit_sequence, noisy_bits) if b != nb) / len(bit_sequence)
    BER = None if BER == 0 else BER

    return noisy_symbol_blocks, SER, BER

def SER_BER_4QAM_MLSE_STATIC(bit_sequence: BIT_SEQUENCE_TYPE, block_size: int, random_values_for_symbols: RANDOM_VALUES_SYMBOLS_TYPE, snr: float) -> Tuple[SYMBOL_BLOCKS_TYPE, SER_TYPE, BER_TYPE]:
    """
    This function uses the given bit sequence, block size, SNR value and random values to perform a simulation that:
        - converts the bit sequence into a symbol sequence using 4QAM
        - splits the symbol sequence into blocks with the given block size
        - add the appended symbols to each sequence in each block. For 4QAM the appended symbols are [0.7071067811865475+0.7071067811865475j, 0.7071067811865475+0.7071067811865475j]
        - add noise to the symbol sequence including the appended symbols in each block using the equation in the practical guide and the static impulse response and SNR.
        - save the noisy symbol sequence from each block which should all be block size + 2 long, thus including the symbols that was appended.
        - perform the MLSE algorithm on each noisy symbol sequence in the blocks to get the corresponding transmitted sequences excluding the appended symbols.
        - combine the blocks of transmitted sequences into a single symbol sequence
        - calculate the SER and BER by comparing the given sequences and transmitted sequences
        - return the blocks of noisy symbols, the SER value scale to log10, and the BER value scale to log10

    parameters:
        bit_sequence -> type <class 'list'> : List containing int items which represents the bits for example: [0, 1, 1, 1, 0, 0, 1, 1]  
        block_size -> type <class 'int'> : An integer indicating the amount of symbols that should be within each block
        random_values_for_symbols -> type <class 'list'> : A list containing lists containing lists. Thus the variable has three index values random_values_for_symbols[0][0][0] and returns a float.
            These random values are used to calculate the added noise according to equation 1 given in the practical guide.
            The first index indicates the corresponding block. The second index indicate the corresponding symbol in that block. The third index indicate the real (0) or imaginary (1) component for which the float value should be used.
            Thus -> random_values_for_symbols[5][84][0] -> is the float that should be used for the real component kappa value to calculate the noise for the 85th symbol in the 6th block (note: the first block is index 0).
        snr -> type <class 'float'> : A float value giving the SNR value that should be used for calculating the sigma value used in the noise equation.

    returns:
        noisy_symbol_blocks -> type <class 'list'> : List containing lists. Each list entry contains a sequence of <class "complex"> which represents the symbols after noise was added for each block
          Example, blocks of size 3 :
            [
                [(-0.262+0.757j), ( 0.071+0.161j), (0.255-0.333j)],
                [( 0.247-0.471j), (-0.668-0.256j), (0.032-0.345j)],
                [(-0.256-0.217j), ( 0.577-0.327j), (0.347-0.359j)],
                [(-0.091+0.641j), ( 0.535+0.101j), (0.203+0.614j)],
            ]
        SER -> type <class 'float'> or <class 'NoneType'> : Returns the SER value (not scaled) or returns the NoneType (return None) if the amount of errors is zero. 
        BER -> type <class 'float'> or <class 'NoneType'> : Returns the BER value (not scaled) or returns the NoneType (return None) if the amount of errors is zero. 
            
    """
    noisy_symbol_blocks = []
    SER = 0
    BER = 0

    symbols = assist_bit_to_symbol(bit_sequence, MAP_4QAM)  
    Blocks = assist_split_symbols_into_blocks(symbols, block_size)  

    CIR = [complex(0.29, 0.98), complex(0.73, -0.24), complex(0.21, 0.91)]

    fbit = m.log2(len(MAP_4QAM))
    sigma = 1 / (m.sqrt(pow(10, (0.1 * snr)) * fbit))

    for i in range(len(Blocks)):
        temp = []
        temp = Blocks[i] 
        noisy = []
        for z in range(len(temp)):
            temp = [complex(0.7071067811865475, 0.7071067811865475), complex(0.7071067811865475, 0.7071067811865475)]
            temp.extend(Blocks[i])
            noisy.append(temp[z + 2] * CIR[0] + temp[z + 1] * CIR[1] + temp[z] * CIR[2] + (sigma * (complex(random_values_for_symbols[i][z][0], random_values_for_symbols[i][z][1]) / m.sqrt(2)))) 
        noisy_symbol_blocks.append(noisy)

    newBlocks = []
    for n in range(len(noisy_symbol_blocks)):
        temp = noisy_symbol_blocks[n]
        temp.append(complex(0.7071067811865475, 0.7071067811865475))
        temp.append(complex(0.7071067811865475, 0.7071067811865475))
        newBlocks.append(MLSE_4QAM_BLOCK(temp, CIR))

    correctedSymbols = assist_combine_blocks_into_symbols(newBlocks)

    SER = 0
    BER = 0
    for k in range(len(symbols)):
        if symbols[k] != correctedSymbols[k]:
            SER = SER + 1
    SER = SER / len(symbols)
    SER = None if SER == 0 else SER

    noisyBits = assist_symbol_to_bit(correctedSymbols, MAP_4QAM)

    for l in range(len(bit_sequence)):
        if bit_sequence[l] != noisyBits[l]:
            BER = BER + 1
    BER = BER / len(bit_sequence)
    BER = None if BER == 0 else BER

    return noisy_symbol_blocks, SER, BER

def SER_BER_4QAM_MLSE_DYNAMIC(bit_sequence: BIT_SEQUENCE_TYPE, block_size: int, random_values_for_symbols: RANDOM_VALUES_SYMBOLS_TYPE, random_values_for_CIR: RANDOM_VALUES_CIR_TYPE, snr: float) -> Tuple[SYMBOL_BLOCKS_TYPE, SER_TYPE, BER_TYPE]:
    """
    This function uses the given bit sequence, block size, SNR value and random values to perform a simulation that:
        - converts the bit sequence into a symbol sequence using BPSK
        - splits the symbol sequence into blocks with the given block size
        - add the appended symbols to each sequence in each block. For BPSK the appended symbols are [0.7071067811865475+0.7071067811865475j, 0.7071067811865475+0.7071067811865475j]
        - calculate a channel impulse response for each block using the dynamic impulse response equation in the practical guide
        - add noise to the symbol sequence including the appended symbols in each block using the equation in the practical guide and the static impulse response and SNR.
        - save the noisy symbol sequence from each block which should all be block size + 2 long, thus including the symbols that was appended.
        - perform the MLSE algorithm on each noisy symbol sequence in the blocks to get the corresponding transmitted sequences excluding the appended symbols.
        - combine the blocks of transmitted sequences into a single symbol sequence
        - calculate the SER and BER by comparing the given sequences and transmitted sequences
        - return the blocks of noisy symbols, the SER value scale to log10, and the BER value scale to log10

    parameters:
        bit_sequence -> type <class 'list'> : List containing int items which represents the bits for example: [0, 1, 1, 1, 0, 0, 1, 1]  
        block_size -> type <class 'int'> : An integer indicating the amount of symbols that should be within each block
        random_values_for_symbols -> type <class 'list'> : A list containing lists containing lists. Thus the variable has three index values random_values_for_symbols[0][0][0] and returns a float.
            These random values are used to calculate the added noise according to equation 1 given in the practical guide.
            The first index indicates the corresponding block. The second index indicate the corresponding symbol in that block. The third index indicate the real (0) or imaginary (1) component for which the float value should be used.
            Thus -> random_values_for_symbols[5][84][0] -> is the float that should be used for the real component kappa value to calculate the noise for the 85th symbol in the 6th block (note: the first block is index 0).
        random_values_for_CIR -> type <class 'list'> : A list containing lists containing lists. Thus the variable has three index values random_values_for_CIR[0][0][0] and returns a float.
            These random values are used to calculate the dynamic impulse response using equation 4 in the practical guide
            The first index indicates the corresponding block. The second index indicate the corresponding coeficient (c0 - [0], c1 = [1], c2 - [2]). The third index indicates the real (0) or imaginary (1) component for which the float value should be used.
            Thus -> random_values_for_CIR[7][2][1] -> is the float that should be used for the imaginary component kappa value to calculate the c2 coeficient for the 8th block (note: the first block is index 0).
        snr -> type <class 'float'> : A float value giving the SNR value that should be used for calculating the sigma value used in the noise equation.

    returns:
        noisy_symbol_blocks -> type <class 'list'> : List containing lists. Each list entry contains a sequence of <class "complex"> which represents the symbols after noise was added for each block
          Example, blocks of size 3 :
            [
                [(-0.262+0.757j), ( 0.071+0.161j), (0.255-0.333j)],
                [( 0.247-0.471j), (-0.668-0.256j), (0.032-0.345j)],
                [(-0.091+0.641j), ( 0.535+0.101j), (0.203+0.614j)],
                [(-0.256-0.217j), ( 0.577-0.327j), (0.347-0.359j)],
            ]
        SER -> type <class 'float'> or <class 'NoneType'> : Returns the SER value (not scaled) or returns the NoneType (return None) if the amount of errors is zero. 
        BER -> type <class 'float'> or <class 'NoneType'> : Returns the BER value (not scaled) or returns the NoneType (return None) if the amount of errors is zero.  
            
    """
    noisy_symbol_blocks = []
    SER = 0
    BER = 0

    symbols = assist_bit_to_symbol(bit_sequence, MAP_4QAM) 
    Blocks = assist_split_symbols_into_blocks(symbols, block_size)  
    CIR = []  

    fbit = m.log2(len(MAP_BPSK))
    sigma = 1 / (m.sqrt(pow(10, (0.1 * snr)) * fbit))

    for i in range(len(Blocks)):
        temp = []
        temp = Blocks[i]
        noisy = []

        c0 = (complex(random_values_for_CIR[i][0][0], random_values_for_CIR[i][0][1])) / m.sqrt(6)
        c1 = (complex(random_values_for_CIR[i][1][0], random_values_for_CIR[i][1][1])) / m.sqrt(6)
        c2 = (complex(random_values_for_CIR[i][2][0], random_values_for_CIR[i][2][1])) / m.sqrt(6)
        CIR = [c0, c1, c2]
        for z in range(len(temp)):
            temp = [complex(1, 0), complex(1, 0)]
            temp.extend(Blocks[i])
            noisy.append(temp[z + 2] * CIR[0] + temp[z + 1] * CIR[1] + temp[z] * CIR[2] + (sigma * (complex(random_values_for_symbols[i][z][0], random_values_for_symbols[i][z][1]) / m.sqrt(2))))

        noisy_symbol_blocks.append(noisy)

    newBlocks = []
    Tosend = noisy_symbol_blocks.copy()
    itest = 0
    for n in Tosend:
        c0 = (complex(random_values_for_CIR[itest][0][0], random_values_for_CIR[itest][0][1])) / m.sqrt(6)
        c1 = (complex(random_values_for_CIR[itest][1][0], random_values_for_CIR[itest][1][1])) / m.sqrt(6)
        c2 = (complex(random_values_for_CIR[itest][2][0], random_values_for_CIR[itest][2][1])) / m.sqrt(6)
        itest += 1
        CIR = [c0, c1, c2]
        n.append(complex(0.7071067811865475, 0.7071067811865475))
        n.append(complex(0.7071067811865475, 0.7071067811865475))
        newBlocks.append(MLSE_4QAM_BLOCK(n, CIR))

    correctedSymbols = assist_combine_blocks_into_symbols(newBlocks)

    SER = 0
    BER = 0
    for k in range(len(symbols)):
        if symbols[k] != correctedSymbols[k]:
            SER = SER + 1
    SER = SER / len(symbols)
    SER = None if SER == 0 else SER

    noisyBits = assist_symbol_to_bit(correctedSymbols, MAP_4QAM)

    for l in range(len(bit_sequence)):
        if bit_sequence[l] != noisyBits[l]:
            BER = BER + 1
    BER = BER / len(bit_sequence)
    BER = None if BER == 0 else BER

    return noisy_symbol_blocks, SER, BER



####### DO NOT EDIT #######
if __name__ == "__main__" :

    evaluate()
####### DO NOT EDIT #######