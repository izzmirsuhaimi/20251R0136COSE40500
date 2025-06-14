import itertools

def vigenere_decrypt(ciphertext, key):
    decrypted_text = []
    key_length = len(key)
    key_indices = [ord(k.upper()) - ord('A') for k in key]  # convert key to indices (0-25)
    for i, char in enumerate(ciphertext):
        if char.isalpha():
            shift = key_indices[i % key_length]  # determine the shift from the key
            char_base = ord('A') if char.isupper() else ord('a')
            decrypted_char = chr((ord(char) - char_base - shift) % 26 + char_base)
            decrypted_text.append(decrypted_char)
        else:
            decrypted_text.append(char)
    return ''.join(decrypted_text)

def iterate_keys():
    first_letter = 'F'
    alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    combinations = itertools.product(alphabet, repeat=2)
    keys = [first_letter + ''.join(combo) for combo in combinations]
    return keys

ciphertext = 'FFJFSFPYRUSBZLCFMFBIEIMFFZR'
keys = iterate_keys()
for k in keys:
    #print(k) # uncomment this to look at key
    decrypted_text = vigenere_decrypt(ciphertext, k)
    if (decrypted_text[0] == 'A'):
        print(decrypted_text)