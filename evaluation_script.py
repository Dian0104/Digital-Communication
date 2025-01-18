import cmath as cm
import numpy as np

#########################################################################
#                       Assisting functions                             #
#########################################################################

def convert_message_to_bits(message: str) -> list:
    """
    Converts a message string to a sequence of bits using ASCII table. Each letter produces 8 bits / 1 byte

    parameters:
        message -> type <class 'str'> : A string containing text

    returns:
        bit_sequence -> type <class 'list'> : List containing int items which represents the bits for example: [0, 1, 1]   
    """
    ascii_codes = [ord(character) for character in message]
    binary_representations = [format(code, '08b') for code in ascii_codes]
    bit_list = [int(bit) for binary in binary_representations for bit in binary]

    return bit_list



#########################################################################
#                       Question 1: Lempel-Ziv                          #
#########################################################################

def lempel_ziv_calculate_dictionary_table(bit_sequence: list) -> list:
    """
    Uses a sequence of bits to determine the dictionary table which can be used to compress a sequence of bits

    parameters:
        bit_sequence -> type <class 'list'> : List containing int items which represents the bits for example: [0, 1, 1] 

    returns:
        lempel_ziv_dictionary_table -> type <class 'list'> : A list containing lists. Each list entry contains three bit sequences, the dictionary location, the dictionary phrase, and the codeword, in that order.
            For example, the first few rows in the lecture notes:
            
            [
                [[0, 0, 0, 0, 1], [1], [0, 0, 0, 0, 0, 1]],
                [[0, 0, 0, 1, 0], [0], [0, 0, 0, 0, 0, 0]],
                [[0, 0, 0, 1, 1], [1, 0], [0, 0, 0, 0, 1, 0]],     
                [[0, 0, 1, 0, 0], [1, 1], [0, 0, 0, 0, 1, 1]],
            ]
            
            Do not sort the array
    """

    unique_sequences = []
    current_buffer = ''

    for bit in bit_sequence:
        current_buffer += str(bit)
        if current_buffer not in unique_sequences:
            unique_sequences.append(current_buffer)
            current_buffer = ''

    sequence_length = int(np.ceil(np.log2(len(unique_sequences))))
    pattern_format = '{:0>' + str(sequence_length) + '}'
    
    location_indices = []
    for i in range(len(unique_sequences)):
        binary_value = bin(i + 1)[2:]
        location_indices.append(pattern_format.format(binary_value))

    encoded_patterns = []

    for seq in unique_sequences:
        prefix = seq[:-1]
        suffix = seq[-1:]
        if not prefix:
            encoded_patterns.append(pattern_format.format('0') + suffix)
        else:
            encoded_patterns.append(location_indices[unique_sequences.index(prefix)] + suffix)

    lempel_ziv_dictionary_table = []
    for loc, seq, enc in zip(location_indices, unique_sequences, encoded_patterns):
        lempel_ziv_dictionary_table.append([[int(num) for num in loc], [int(num) for num in seq], [int(num) for num in enc]])

    return lempel_ziv_dictionary_table

def lempel_ziv_compress_bit_sequence(bit_sequence: list, lempel_ziv_dictionary_table: list) -> list:
    """
    Compresses a sequence of bits using the lempel-ziv algorithm and the lempel ziv codewords in the lempel ziv dictionary table

    parameters:
        bit_sequence -> type <class 'list'> : List containing int items which represents the bits for example: [0, 1, 1] 
        lempel_ziv_dictionary_table -> type <class 'list'> : A list containing lists. Each list entry contains three bit sequences, the dictionary location, the dictionary phrase, and the codeword, in that order.
            See example at function lempel_ziv_calculate_dictionary_table

    returns:
        compressed_bit_sequence -> type <class 'list'> : List containing int items which represents the bits for example: [0, 1, 1] 
    """
     
    pattern_list = [entry[1] for entry in lempel_ziv_dictionary_table]
    encoded_values = [entry[2] for entry in lempel_ziv_dictionary_table]
    compressed_output = []

    sequence = list(bit_sequence)
    max_pattern_length = max(len(pattern) for pattern in pattern_list)

    while sequence:
        found_match = False
        for length in range(max_pattern_length, 0, -1):
            subsequence = sequence[:length]
            if subsequence in pattern_list:
                compressed_output.extend(encoded_values[pattern_list.index(subsequence)])
                sequence = sequence[length:]
                found_match = True
                break
        
        if not found_match:
            break

    return compressed_output
   

    
   

def lempel_ziv_decompress_bit_sequence(compressed_bit_sequence: list, lempel_ziv_dictionary_table: list) -> list:
    """
    Decompress a sequence of bits using the lempel-ziv algorithm and the lempel-ziv codewords in the lempel-ziv dictionary table

    parameters:
        compressed_bit_sequence -> type <class 'list'> : List containing int items which represents the bits for example: [0, 1, 1] 
        lempel_ziv_dictionary_table -> type <class 'list'> : A list containing lists. Each list entry contains three bit sequences, the dictionary location, the dictionary phrase, and the codeword, in that order.
            See example at function lempel_ziv_calculate_dictionary_table

    returns:
        bit_sequence -> type <class 'list'> : List containing int items which represents the bits for example: [0, 1, 1] 
    """
    bit_sequence = []
    pattern_list = [entry[1] for entry in lempel_ziv_dictionary_table]
    encoded_sequences = [entry[2] for entry in lempel_ziv_dictionary_table]
    codeword_length = len(lempel_ziv_dictionary_table[0][2])

    for i in range(0, len(compressed_bit_sequence), codeword_length):
        segment = compressed_bit_sequence[i:i + codeword_length]
        for bit in pattern_list[encoded_sequences.index(segment)]:
            bit_sequence.append(bit)

    return bit_sequence

#########################################################################
#                       Question 2: Huffman coding                      #
#########################################################################

def huffman_get_mapping_codes() -> list:
    """
    Returns the mapping codes generated using a huffman coding tree, which can be used to compress a message 

    parameters:
        None -> The tree can be calculated by hand and this function just returns the result

    returns:
        huffman_mapping -> type <class 'list'> : A list containing lists. Each list entry contains a string and a bit sequence, the letter as a string, and the corresponding bit sequence.
            For example, the huffman mapping codes for example 1 in the lecture notes:

            [
                ['x_1', [0, 0]],
                ['x_2', [0, 1]],
                ['x_3', [1, 0]],
                ['x_4', [1, 1, 0]],
                ['x_5', [1, 1, 1, 0]],
                ['x_6', [1, 1, 1, 1, 0]],
                ['x_7', [1, 1, 1, 1, 1]]
            ]
    """

    a = [
        [' ',[1,0]],
        ['a',[0,0,0,0]],
        ['c',[0,0,1,1]],
        ['d',[0,1,1,0]],
        ['b',[1,1,0,0]],
        ['e',[0,0,0,1,0]],
        ['j',[0,0,0,1,1]],
        ['n',[0,0,1,0,1]],
        ['k',[0,1,0,0,0]],
        ['q',[0,1,0,1,0]],
        ['r',[0,1,0,1,1]],
        ['i',[0,1,1,1,0]],
        ['u',[0,1,1,1,1]],
        ['s',[1,1,0,1,0]],
        ['h',[1,1,1,0,0]],
        ['t',[1,1,1,0,1]],
        ['y',[1,1,1,1,0]],
        ['f',[0,0,1,0,0,0]],
        ['z',[0,0,1,0,0,1]],
        ['x',[0,1,0,0,1,0]],
        ['m',[0,1,0,0,1,1]],
        ['o',[1,1,0,1,1,0]],
        ['v',[1,1,0,1,1,1]],
        ['p',[1,1,1,1,1,0]],
        ['l',[1,1,1,1,1,1,0]],
        ['w',[1,1,1,1,1,1,1,0]],
        ['g',[1,1,1,1,1,1,1,1,0]],
        ['.',[1,1,1,1,1,1,1,1,1]]
    ]

    return a

def huffman_compress_message(message: str, huffman_mapping: list) -> list:
    """
    Compresses a text message using the huffman mapping codes generated in function huffman_get_mapping_codes and generates a sequence of bits. Assume input consists of only characters within the huffman mapping codes.

    parameters:
        message -> type <class 'str'> : A string containing text
        huffman_mapping -> type <class 'list'> : A list containing lists. Each list entry contains a string and a bit sequence, the letter as a string, and the corresponding bit sequence.
            See example at function huffman_get_mapping_codes

    returns:
        bit_sequence -> type <class 'list'> : List containing int items which represents the bits for example: [0, 1, 1] 
    """

    bsequence = []
    lowercase_message = message.lower()

    for character in lowercase_message:
        code = None
        for entry in huffman_mapping:
            if entry[0] == character:
                code = entry[1]
                break

        if code is not None:
            bsequence.extend(code)

    return bsequence

def huffman_decompress_bit_sequence(bit_sequence: list, huffman_mapping: list) -> list:
    """
    Decompresses a text message using the huffman mapping codes generated in function huffman_get_mapping_codes and generates single text message.

    parameters:
        bit_sequence -> type <class 'list'> : List containing int items which represents the bits for example: [0, 1, 1] 
        huffman_mapping -> type <class 'list'> : A list containing lists. Each list entry contains a string and a bit sequence, the letter as a string, and the corresponding bit sequence.
            See example at function huffman_get_mapping_codes

    returns:
        message -> type <class 'str'> : A string containing text
    """

    decoded_message = ""
    accumulated_bits = []

    for bit in bit_sequence:
        accumulated_bits.append(bit)
        for entry in huffman_mapping:
            if accumulated_bits == entry[1]:
                decoded_message += entry[0]
                accumulated_bits = []
                break

    return decoded_message

#########################################################################
#                       Question 3: Simulation Platform                 #
#########################################################################

def modulation_get_symbol_mapping(scheme: str) -> list:
    """
    Returns the bit to symbol mapping for the given scheme. Returns NoneType if provided scheme name is not available.
    Required scheme implementations: "BPSK", "4QAM", "8PSK", "16QAM".    

    parameters:
        scheme -> type <class 'str'> : A string containing the name of the scheme for which a bit to symbol mapping needs to be returned.

    returns:
        bit_to_symbol_map -> type <class 'list'> : A list containing lists. Each list entry contains a complex number and a bit sequence, representing the symbol, and the corresponding bit sequence that maps to it.
        For example:
            [
                [ 1+1j,    [0, 0] ],
                [ -1+1j,   [0, 1] ],
                [ -1-1j,   [1, 1] ],
                [ 1-1j,    [1, 0] ]
            ]
    """
    symbol_mapping = []

    if scheme == 'BPSK':
        symbol_mapping = [
            [1, [1]],
            [-1, [0]]
        ]
    elif scheme == '4QAM':
        symbol_mapping = [
            [complex(1, 1) / np.sqrt(2), [0, 0]],
            [complex(-1, 1) / np.sqrt(2), [0, 1]],
            [complex(-1, -1) / np.sqrt(2), [1, 1]],
            [complex(1, -1) / np.sqrt(2), [1, 0]]
        ]
    elif scheme == '8PSK':
        symbol_mapping = [
            [complex(-1, -1) / np.sqrt(2), [0, 0, 0]],
            [-1, [0, 0, 1]],
            [complex(-1, 1) / np.sqrt(2), [0, 1, 1]],
            [complex(0, 1), [0, 1, 0]],
            [complex(1, 1) / np.sqrt(2), [1, 1, 0]],
            [1, [1, 1, 1]],
            [complex(1, -1) / np.sqrt(2), [1, 0, 1]],
            [complex(0, -1), [1, 0, 0]]
        ]
    elif scheme == '16QAM':
        symbol_mapping = [
            [complex(-0.5, 0.5) * np.sqrt(2), [0, 0, 0, 0]],
            [complex(-1/6, 0.5) * np.sqrt(2), [0, 1, 0, 0]],
            [complex(1/6, 0.5) * np.sqrt(2), [1, 1, 0, 0]],
            [complex(0.5, 0.5) * np.sqrt(2), [1, 0, 0, 0]],
            [complex(-0.5, 1/6) * np.sqrt(2), [0, 0, 0, 1]],
            [complex(-1/6, 1/6) * np.sqrt(2), [0, 1, 0, 1]],
            [complex(1/6, 1/6) * np.sqrt(2), [1, 1, 0, 1]],
            [complex(0.5, 1/6) * np.sqrt(2), [1, 0, 0, 1]],
            [complex(-0.5, -1/6) * np.sqrt(2), [0, 0, 1, 1]],
            [complex(-1/6, -1/6) * np.sqrt(2), [0, 1, 1, 1]],
            [complex(1/6, -1/6) * np.sqrt(2), [1, 1, 1, 1]],
            [complex(0.5, -1/6) * np.sqrt(2), [1, 0, 1, 1]],
            [complex(-0.5, -0.5) * np.sqrt(2), [0, 0, 1, 0]],
            [complex(-1/6, -0.5) * np.sqrt(2), [0, 1, 1, 0]],
            [complex(1/6, -0.5) * np.sqrt(2), [1, 1, 1, 0]],
            [complex(0.5, -0.5) * np.sqrt(2), [1, 0, 1, 0]]
        ]
    elif scheme == "QPSK":
        symbol_mapping = [
            [0 + 0j, [0, 0]],
            [0 + 1j, [0, 1]],
            [-1 + 0j, [1, 1]],
            [-1 - 1j, [1, 0]]
        ]

    return symbol_mapping

def modulation_map_bits_to_symbols(bit_to_symbol_map: list, bit_sequence: list) -> list:
    """
    Returns a sequence of symbols that corresponds to the provided sequence of bits using the bit to symbol map that respresent the modulation scheme

    parameters:
        bit_to_symbol_map -> type <class 'list'> : A list containing lists. Each list entry contains a complex number and a bit sequence, representing the symbol, and the corresponding bit sequence that maps to it.
            See example at function modulation_get_symbol_mapping
        bit_sequence -> type <class 'list'> : List containing int items which represents the bits for example: [0, 1, 1] 
        
    returns:
        symbol_sequence -> type <class 'list'> : List containing complex items (<class "complex">) which represents the symbols for example: [1+1j, 2+2j]

    """

    decoded_sequence = []
    modification = ""
    total_elements = 0

    if len(bit_to_symbol_map[0][1]) == 1:
        total_elements = len(bit_sequence)
        decoded_sequence = np.array(bit_sequence).reshape(total_elements, 1).tolist()
    elif len(bit_to_symbol_map[0][1]) == 2:
        total_elements = len(bit_sequence) // 2
        decoded_sequence = np.array(bit_sequence).reshape(total_elements, 2).tolist()
    elif len(bit_to_symbol_map[0][1]) == 3:
        total_elements = len(bit_sequence) // 3
        decoded_sequence = np.array(bit_sequence).reshape(total_elements, 3).tolist()
    elif len(bit_to_symbol_map[0][1]) == 4:
        total_elements = len(bit_sequence) // 4
        decoded_sequence = np.array(bit_sequence).reshape(total_elements, 4).tolist()

    symbol_sequence = [0] * total_elements
    for idx, seq in enumerate(decoded_sequence):
        for map_entry in bit_to_symbol_map:
            if seq == map_entry[1]:
                symbol_sequence[idx] = map_entry[0]

    return symbol_sequence

def modulation_map_symbols_to_bits(bit_to_symbol_map: list, symbol_sequence: list) -> list:
    """
    Returns a sequence of bits that corresponds to the provided sequence of symbols containing noise using the bit to symbol map that respresent the modulation scheme and the euclidean distance

    parameters:
        bit_to_symbol_map -> type <class 'list'> : A list containing lists. Each list entry contains a complex number and a bit sequence, representing the symbol, and the corresponding bit sequence that maps to it.
            See example at function modulation_get_symbol_mapping
        symbol_sequence -> type <class 'list'> : List containing complex items (<class "complex">) which represents the symbols for example: [1+1j, 2+2j]
        
    returns:
        bit_sequence -> type <class 'list'> : List containing int items which represents the bits for example: [0, 1, 1] 

    """
    modulation_scheme = ""
    if len(bit_to_symbol_map[0][1]) == 1:
        modulation_scheme = "BPSK"
    elif len(bit_to_symbol_map[0][1]) == 2:
        modulation_scheme = "4QAM"
    elif len(bit_to_symbol_map[0][1]) == 3:
        modulation_scheme = "8PSK"
    elif len(bit_to_symbol_map[0][1]) == 4:
        modulation_scheme = "16QAM"

    decoded_bits = [0] * len(symbol_sequence)
    for idx, symbol in enumerate(symbol_sequence):
        min_distance = float('inf')
        closest_index = 0
        for map_idx, map_entry in enumerate(bit_to_symbol_map):
            real_diff = map_entry[0].real - symbol.real
            imag_diff = map_entry[0].imag - symbol.imag
            distance = np.sqrt(real_diff**2 + imag_diff**2)
            if distance < min_distance:
                closest_index = map_idx
                min_distance = distance
        decoded_bits[idx] = bit_to_symbol_map[closest_index][1]

    flattened_bits = sum(decoded_bits, [])
    return flattened_bits

def complex_distance(a: complex, b: complex) -> float:
    return np.sqrt((a.real - b.real)**2 + (a.imag - b.imag)**2)

def modulation_determine_SER_and_BER(bit_sequence: list, 
                                     symbol_sequence: list, 
                                     bit_to_symbol_map: list, 
                                     gaussian_random_values: list, 
                                     SNR_range: list) -> tuple:
    
    """
    Returns a range of SER and BER values over the given SNR range using the supplied sequence of bits and symbols and noise calculated using the gaussian random values.

    parameters:
        bit_sequence -> type <class 'list'> : List containing int items which represents the bits for example: [0, 1, 1] 
        symbol_sequence -> type <class 'list'> : List containing complex items (<class "complex">) which represents the symbols for example: [1+1j, 2+2j]
            This symbol list, is the result of the function modulation_map_bits_to_symbols using the bit_sequence and bit_to_symbol_map given for this function
        bit_to_symbol_map -> type <class 'list'> : A list containing lists. Each list entry contains a complex number and a bit sequence, representing the symbol, and the corresponding bit sequence that maps to it.
            See example at function modulation_get_symbol_mapping
        gaussian_random_values -> type <class 'list'> : A list containing lists. Each list entry contains two floats, both random gaussian distributed values.
            These random values are used to calculate the added noise according to the equation given in the practical guide. The first number should be used for the real component, and the second number should be used for the imaginary component.
        SNR_range -> type <class 'list'> : A list containing all the SNR values for which a SER and BER should be calculated for

    returns:
        noisy_symbol_sequence -> type <class 'list'> : A list containing lists. Each list entry contains complex items (<class "complex">) which represents the symbols for example: [1+1j, 2+2j]
            Since multiple SNR values are provided, a list of multiple symbol sequences should be returned for each corresponding SNR value in the SNR_range variable.
        noisy_bit_sequence - type <class 'list'> : A list containing lists. Each list entry contains int items which represents the bits for example: [0, 1, 1]
            Since multiple SNR values are provided, a list of multiple bit sequences should be returned for each corresponding SNR value in the SNR_range variable.
        SER_results -> <class 'list'> : A list containing the SER as float for the corresponding SNR value in the same index position in the SNR_range list, for example [-2.5, -5.31]
            The result should be scaled using log(SER) (base 10)
        BER_results -> <class 'list'> : A list containing the BER as float for the corresponding SNR value in the same index position in the SNR_range list, for example [-2.5, -5.31]
            The result should be scaled using log(BER) (base 10)

    Additional Explanation:

    As previously, the bit_sequence variable is a list of 1 and 0 integers. 
    These bits were then converted to a list of symbols using the function
    modulation_map_bits_to_symbols, and the results were stored in the
    symbol_sequence variable. As normal the mapping between symbols and bits
    are given in the variable bit_to_symbol_map, obtained from function 
    modulation_get_symbol_mapping. 

    Only one bit sequence and symbol sequence are provided to the function. The
    same sequences should be used to add noise for the different SNR values.

    The gaussian_random_values variable are a list of zero mean, unity variance 
    Gaussian random numbers. Each entry corresponds to a symbol in the 
    symbol_sequence variable, meaning that the noise that should be added to 
    symbol_sequence[4] should be gaussian_random_values[4]. Each entry in the 
    list consists of two random numbers. The first random number should be used
    as the real number "n_k^(i)" from equation 1 in the practical guide. The 
    second number in the entry should be used as the imaginary number "n_k^(q)"
    from equation 1 in the practical guide. 

    For example, suppose the symbol list is as [(1 + 1j), ...] and the 
    gaussian_random_values is as [ [1.23, -0.5] , ...]. For the first symbol
    the two noise values for the real and imaginary component are 1.23 and -0.5
    respectively, thus the symbol with noise will then be:
                (1 + 1j) + sigma * (1.23 - 0.5j) / sqrt(2) 

    Remember that the equation changes for BPSK, and only the first value is used

    The list of symbols that were produced by adding the noise to the 
    symbol_sequence is the noisy_symbols variable that should be returned. Since 
    multiple SNR ranges are computed, multiple lists of noisy symbols should be 
    returned within one list in the same position as the SNR value used for example:

        SNR_range =             [           1,                     2,                   3          ]
        noisy_symbol_sequence = [ [(1.23 - 0.5j), ...], [(0.21 - 1.2j), ...], [(0.53 + 0.2j), ...] ]

    The same is done for the noisy_bit_sequence

        noisy_bit_sequence =    [ [0, 1, 0, 0, 1, ...], [0, 1, 1, 0, 1, ...], [0, 1, 0, 1, 1, ...] ]

    Finally, the SER and BER for the different SNR values should be computed and a 
    list for each is returned. Remember to scale it using log. If the value for SER and BER is zero,
    which means log10 can not be computed, then return None, type NoneType.

        SER_results =           [         -24,                   -56,                 -120         ]
        BER_results =           [         -32,                   -73,                 -185         ]

    """

    noisy_symbol_sequence = []
    noisy_bit_sequence = []
    SER_results = []
    BER_results = []

    for SNR in SNR_range:
        noisy_symbol = []
        sigma = 10.0 ** (SNR * 0.1)
        sigma *= 2.0 * int(np.log2(len(bit_to_symbol_map)))
        sigma = 1.0 / np.sqrt(sigma)

        for symbol, noise in zip(symbol_sequence, gaussian_random_values):
            if len(noise) > 0.5:
                noisy_symbol.append(symbol + (sigma * (noise[0] + noise[1] * 1j) / np.sqrt(2)))
            else:
                noisy_symbol.append(symbol + (sigma * noise[0] / np.sqrt(2)))

        noisy_bit = modulation_map_symbols_to_bits(bit_to_symbol_map, noisy_symbol)

        # Calculate SER
        num_symbols = len(symbol_sequence)
        symbol_errors = sum(not np.isclose(symbol, map_symbol[0], atol=1e-8) 
                            for symbol, map_symbol in zip(symbol_sequence, 
                                                          [bit_to_symbol_map[np.argmin([complex_distance(symbol, entry[0]) for entry in bit_to_symbol_map])] for symbol in noisy_symbol]))

        SER = symbol_errors / num_symbols
        SER_results.append(None if SER == 0 else np.log10(SER))

        # Calculate BER
        num_bits = len(bit_sequence)
        bit_errors = sum(bit1 != bit2 for bit1, bit2 in zip(bit_sequence, noisy_bit))

        BER = bit_errors / num_bits
        BER_results.append(None if BER == 0 else np.log10(BER))

        noisy_symbol_sequence.append(noisy_symbol)
        noisy_bit_sequence.append(noisy_bit)

    return noisy_symbol_sequence, noisy_bit_sequence, SER_results, BER_results