import itertools

def decrypt_caesar_cipher(ciphertext, shift):
    decrypted_text = ""
    for char in ciphertext:
        if char.isalpha():
            decrypted_char = chr((ord(char) - ord('A') - shift) % 26 + ord('A'))
            decrypted_text += decrypted_char
        else:
            decrypted_text += char
    return decrypted_text

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
    first_letter = 'T'
    alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    combinations = itertools.product(alphabet, repeat=2)
    keys = [first_letter + ''.join(combo) for combo in combinations]
    return keys

def check_positions(plaintext):     # check if the letters at positions 4, 7, 11, 14, and 22 are the same
    positions = [3, 6, 10, 13, 21]
    letters = [plaintext[pos] for pos in positions]
    # All letters should be the same
    return len(set(letters)) == 1

ciphertext = 'YJZAGBATXHVAEVHCZXSOTAIXWZWS'
for i in range(26):
    if i <= 26:
        print(i)
        shift = i
        decrypted_shift = decrypt_caesar_cipher(ciphertext, shift)
        keys = iterate_keys()
        for k in keys:
            #print(k) # uncomment this to look at key
            plain_text = vigenere_decrypt(decrypted_shift, k)
            x = check_positions(plain_text)
            if x is True:
                print(plain_text)
        i += 1