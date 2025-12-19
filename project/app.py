from flask import Flask, render_template, request, jsonify
from collections import Counter
import heapq
import math
import itertools
from decimal import Decimal, getcontext

app = Flask(__name__)

# Set precision for Arithmetic Coding
getcontext().prec = 50

# --- UTILITY FUNCTIONS ---

def get_probabilities(text):
    """Calculates probabilities for Shannon and Shannon-Fano encoding."""
    freq = Counter(text)
    total = len(text)
    symbols = list(freq.keys())
    probs = [freq[char] / total for char in symbols]
    return symbols, probs

def decimal_to_binary(fraction, max_bits=64):
    """Convert decimal fraction to binary string."""
    if fraction == 0:
        return '0' * max_bits
    binary = ''
    x = fraction
    for _ in range(max_bits):
        x *= 2
        if x >= 1:
            binary += '1'
            x -= 1
        else:
            binary += '0'
    return binary

# --- ARITHMETIC CODING ---

def arithmetic_encode(text):
    """Arithmetic Encoding Algorithm"""
    if not text:
        return ""
    
    # 1. Calculate frequency and probability of each symbol
    freq = Counter(text)
    total = len(text)
    
    # Sort symbols for consistent ordering
    symbols = sorted(freq.keys())
    
    # 2. Build cumulative probability ranges
    cumulative_prob = {}
    low_range = Decimal(0)
    
    for symbol in symbols:
        prob = Decimal(freq[symbol]) / Decimal(total)
        cumulative_prob[symbol] = {
            'low': low_range,
            'high': low_range + prob
        }
        low_range += prob
    
    # 3. Encode the text
    low = Decimal(0)
    high = Decimal(1)
    
    for char in text:
        range_width = high - low
        high = low + range_width * cumulative_prob[char]['high']
        low = low + range_width * cumulative_prob[char]['low']
    
    # 4. Choose a value in the final range (midpoint)
    encoded_value = (low + high) / 2
    
    # Store frequency table as well for decoding
    freq_str = str(dict(freq))
    
    return f"{encoded_value}|{freq_str}|{len(text)}"

def arithmetic_decode(data):
    """Arithmetic Decoding Algorithm"""
    try:
        # Parse encoded data
        parts = data.split("|")
        encoded_value = Decimal(parts[0])
        freq = eval(parts[1])
        length = int(parts[2])
        
        # Rebuild cumulative probability ranges
        total = sum(freq.values())
        symbols = sorted(freq.keys())
        
        cumulative_prob = {}
        low_range = Decimal(0)
        
        for symbol in symbols:
            prob = Decimal(freq[symbol]) / Decimal(total)
            cumulative_prob[symbol] = {
                'low': low_range,
                'high': low_range + prob
            }
            low_range += prob
        
        # Decode
        result = []
        value = encoded_value
        
        for _ in range(length):
            # Find which symbol's range contains the value
            for symbol in symbols:
                if cumulative_prob[symbol]['low'] <= value < cumulative_prob[symbol]['high']:
                    result.append(symbol)
                    
                    # Update value for next symbol
                    range_width = cumulative_prob[symbol]['high'] - cumulative_prob[symbol]['low']
                    value = (value - cumulative_prob[symbol]['low']) / range_width
                    break
        
        return "".join(result)
    except Exception as e:
        return f"Arithmetic Decoding Error: {str(e)}"

# --- LZW COMPRESSION ---

def lzw_encode(text):
    """LZW Encoding Algorithm"""
    if not text:
        return ""
    
    # 1. Initialize dictionary with single characters
    dict_size = 256
    dictionary = {chr(i): i for i in range(dict_size)}
    
    # 2. Encode
    result = []
    current_string = ""
    
    for char in text:
        combined = current_string + char
        
        if combined in dictionary:
            current_string = combined
        else:
            # Output code for current_string
            result.append(dictionary[current_string])
            
            # Add combined to dictionary
            dictionary[combined] = dict_size
            dict_size += 1
            
            # Start new string with current char
            current_string = char
    
    # Output code for remaining string
    if current_string:
        result.append(dictionary[current_string])
    
    return ",".join(map(str, result))

def lzw_decode(data):
    """LZW Decoding Algorithm"""
    try:
        # Parse encoded data
        codes = [int(x.strip()) for x in data.split(",")]
        
        # 1. Initialize dictionary with single characters
        dict_size = 256
        dictionary = {i: chr(i) for i in range(dict_size)}
        
        # 2. Decode
        result = []
        
        # Read first code
        previous_code = codes[0]
        result.append(dictionary[previous_code])
        
        for code in codes[1:]:
            if code in dictionary:
                entry = dictionary[code]
            elif code == dict_size:
                # Special case: code not in dictionary yet
                entry = dictionary[previous_code] + dictionary[previous_code][0]
            else:
                return f"LZW Decoding Error: Invalid code {code}"
            
            result.append(entry)
            
            # Add new entry to dictionary
            dictionary[dict_size] = dictionary[previous_code] + entry[0]
            dict_size += 1
            
            previous_code = code
        
        return "".join(result)
    except Exception as e:
        return f"LZW Decoding Error: {str(e)}"

# --- SHANNON-FANO CODING (Based on your Pseudocode) ---

def shannon_fano_encode(text):
    symbols, probabilities = get_probabilities(text)
    n = len(symbols)
    
    # 1. Sort symbols in descending order of probability
    data = sorted(zip(symbols, probabilities), key=lambda x: x[1], reverse=True)
    sorted_symbols = [d[0] for d in data]
    sorted_probs = [d[1] for d in data]
    
    codes = {symbol: "" for symbol in sorted_symbols}

    # Procedure AssignCodes(symbols, start, end)
    def assign_codes(start, end):
        # 3. If start == end: return
        if start == end:
            return
        
        # 4. Find partition point 'k' where:
        # sum(prob[start..k]) is approximately equal to sum(prob[k+1..end])
        total_sum = sum(sorted_probs[start:end+1])
        prefix_sum = 0
        min_diff = float('inf')
        k = start
        
        for i in range(start, end):
            prefix_sum += sorted_probs[i]
            diff = abs((total_sum - prefix_sum) - prefix_sum)
            if diff < min_diff:
                min_diff = diff
                k = i
            else:
                break # Difference is increasing, we found the best k

        # 5. For i in start..k: append '0'
        for i in range(start, k + 1):
            codes[sorted_symbols[i]] += '0'
        
        # 6. For i in k+1..end: append '1'
        for i in range(k + 1, end + 1):
            codes[sorted_symbols[i]] += '1'

        # 7. Recursively call AssignCodes(symbols, start, k)
        assign_codes(start, k)
        # 8. Recursively call AssignCodes(symbols, k+1, end)
        assign_codes(k + 1, end)

    # 2. Call AssignCodes(symbols, start=0, end=n-1)
    if n > 0:
        assign_codes(0, n - 1)
    
    # Generate the encoded bitstream for the original text
    encoded_bits = "".join([codes[char] for char in text])
    return f"{encoded_bits}|{codes}"

def shannon_fano_decode(data):
    """Algorithm ShannonFanoDecode(encoded_bits, code_table)"""
    try:
        # Expects format: bits|code_table_string
        encoded_bits, cb_str = data.split("|")
        code_table = eval(cb_str) # Convert string representation back to dict
        
        # 1. result <- ""
        result = ""
        # 2. buffer <- ""
        buffer = ""
        
        # Create a reverse lookup for matching codes to symbols
        reverse_table = {v: k for k, v in code_table.items()}
        
        # 3. For each bit in encoded_bits:
        for bit in encoded_bits:
            # buffer <- buffer + bit
            buffer += bit
            # If buffer matches any code in code_table:
            if buffer in reverse_table:
                # append corresponding symbol to result
                result += reverse_table[buffer]
                # buffer <- ""
                buffer = ""
        
        # 4. return result
        return result
    except Exception as e:
        return f"Shannon-Fano Decoding Error: {str(e)}"

# --- SHANNON CODING ---
def shannon_encode(text):
    symbols, probabilities = get_probabilities(text)
    sorted_data = sorted(zip(symbols, probabilities), key=lambda x: x[1], reverse=True)

    cumulative = 0
    F = [0]
    for i in range(len(sorted_data) - 1):
        cumulative += sorted_data[i][1]
        F.append(cumulative)

    codebook = {}
    for i, (symbol, prob) in enumerate(sorted_data):
        if prob == 0: continue
        length = math.ceil(-math.log2(prob))
        binary_frac = decimal_to_binary(F[i], length)
        codebook[symbol] = binary_frac[:length]
    
    encoded_bits = "".join([codebook[char] for char in text])
    return f"{encoded_bits}|{codebook}"

def shannon_decode(data):
    try:
        bitstream, cb_str = data.split("|")
        codebook = eval(cb_str)
        reverse_codebook = {v: k for k, v in codebook.items()}
        result = []
        buffer = ''
        for bit in bitstream:
            buffer += bit
            if buffer in reverse_codebook:
                result.append(reverse_codebook[buffer])
                buffer = ''
        return "".join(result)
    except Exception as e:
        return f"Shannon Decoding Error: {str(e)}"

# --- MTF & WMTF LOGIC ---
# (Logic remains unchanged as per your provided code)
def mtf_encode(text, alphabet_input):
    abc = list(alphabet_input) if alphabet_input else list("abcdefghijklmnopqrstuvwxyz")
    for char in reversed(text):
        if char not in abc: abc.insert(0, char)
    symbol_list = list(abc)
    encoded = []
    for c in text:
        pos = symbol_list.index(c)
        encoded.append(pos)
        symbol_list.pop(pos)
        symbol_list.insert(0, c)
    return f"{','.join(map(str, encoded))}|{''.join(abc)}"

def mtf_decode(data, alphabet_input):
    try:
        if "|" in data:
            idx_str, abc_str = data.split("|")
            indices = [int(x.strip()) for x in idx_str.split(",")]
            symbol_list = list(abc_str)
        else:
            indices = [int(x.strip()) for x in data.split(",")]
            symbol_list = list(alphabet_input) if alphabet_input else list("abcdefghijklmnopqrstuvwxyz")
        res = []
        for i in indices:
            char = symbol_list[i]
            res.append(char)
            symbol_list.pop(i)
            symbol_list.insert(0, char)
        return "".join(res)
    except: return "MTF Error"

def w_mtf_encode(S, L_input, beta):
    L = list(L_input) if L_input else []
    if not L:
        for char in S:
            if char not in L: L.append(char)
    initial_L = "".join(L)
    E = []
    for s in S:
        i = L.index(s)
        E.append(i)
        new_pos = math.floor(i * beta)
        L.pop(i)
        L.insert(new_pos, s)
    return f"{','.join(map(str, E))}|{initial_L}|{beta}"

def w_mtf_decode(data, L_manual, beta_manual):
    try:
        if "|" in data:
            parts = data.split("|")
            E = [int(x) for x in parts[0].split(",")]
            L = list(parts[1])
            beta = float(parts[2])
        else:
            E = [int(x) for x in data.split(",")]
            L = list(L_manual)
            beta = float(beta_manual)
        S = []
        for i in E:
            s = L[i]
            S.append(s)
            new_pos = math.floor(i * beta)
            L.pop(i)
            L.insert(new_pos, s)
        return "".join(S)
    except: return "WMTF Error"

# --- RLE LOGIC ---

def rle_encode(text):
    return "".join([f"{len(list(g))}{k}" for k, g in itertools.groupby(text)])

def rle_decode(text):
    res, i = "", 0
    while i < len(text):
        count = ""
        while i < len(text) and text[i].isdigit():
            count += text[i]; i += 1
        if i < len(text):
            res += text[i] * int(count); i += 1
    return res

class HuffmanNode:
    def __init__(self, symbol, freq):
        self.symbol = symbol
        self.freq = freq
        self.left = None
        self.right = None

    # Define comparison for the priority queue
    def __lt__(self, other):
        return self.freq < other.freq

# --- HUFFMAN CODING ---

def huffman_encode(text):
    if not text: return ""
    
    # 1. Create a priority queue Q of nodes
    freqs = Counter(text)
    priority_queue = [HuffmanNode(char, f) for char, f in freqs.items()]
    heapq.heapify(priority_queue)

    # 3. Repeat until Q has only one node
    while len(priority_queue) > 1:
        a = heapq.heappop(priority_queue)
        b = heapq.heappop(priority_queue)
        
        # Create new internal node z
        z = HuffmanNode(None, a.freq + b.freq)
        z.left = a
        z.right = b
        heapq.heappush(priority_queue, z)

    # 4. Let Root = only remaining node
    root = priority_queue[0]
    
    # 5. Traverse tree to assign codes
    codebook = {}
    def get_codes(node, current_code):
        if not node: return
        if node.symbol is not None:
            codebook[node.symbol] = current_code
        get_codes(node.left, current_code + "0")
        get_codes(node.right, current_code + "1")

    get_codes(root, "")
    
    encoded_bits = "".join([codebook[char] for char in text])
    # We return bits and the codebook for the decoder to use
    return f"{encoded_bits}|{codebook}"

def huffman_decode(data):
    try:
        bitstream, cb_str = data.split("|")
        codebook = eval(cb_str)
        # Reverse the codebook for decoding
        reverse_cb = {v: k for k, v in codebook.items()}
        
        result = ""
        buffer = ""
        # Implementation of your decoding pseudocode
        for bit in bitstream:
            buffer += bit
            if buffer in reverse_cb:
                result += reverse_cb[buffer]
                buffer = ""
        return result
    except Exception as e:
        return f"Huffman Decoding Error: {str(e)}"

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/run", methods=["POST"])
def run_algorithm():
    data = request.get_json()
    text, alphabet = data.get("text", ""), data.get("alphabet", "")
    beta = float(data.get("beta", 0.5))
    algo, mode = data.get("algorithm", ""), data.get("mode", "")

    if algo == "shannon":
        result = shannon_encode(text) if mode == "encode" else shannon_decode(text)
    elif algo == "sf":
        result = shannon_fano_encode(text) if mode == "encode" else shannon_fano_decode(text)
    elif algo == "mtf":
        result = mtf_encode(text, alphabet) if mode == "encode" else mtf_decode(text, alphabet)
    elif algo == "w_mtf":
        result = w_mtf_encode(text, alphabet, beta) if mode == "encode" else w_mtf_decode(text, alphabet, beta)
    elif algo == "rle":
        result = rle_encode(text) if mode == "encode" else rle_decode(text)
    elif algo == "huffman":
        result = huffman_encode(text) if mode == "encode" else huffman_decode(text)
    elif algo == "arithmetic":
        result = arithmetic_encode(text) if mode == "encode" else arithmetic_decode(text)
    elif algo == "lzw":
        result = lzw_encode(text) if mode == "encode" else lzw_decode(text)
    else:
        result = "Algorithm Not Found"

    return jsonify({"result": result})

if __name__ == "__main__":
    app.run(debug=True)
