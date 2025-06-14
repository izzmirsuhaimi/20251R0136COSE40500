def decrypt_caesar_cipher(ciphertext, shift):
    decrypted_text = ""
    for char in ciphertext:
        if char.isalpha():
            decrypted_char = chr((ord(char) - ord('A') - shift) % 26 + ord('A'))
            decrypted_text += decrypted_char
        else:
            decrypted_text += char
    return decrypted_text

#ciphertext = input("Enter the ciphertext: ")
ciphertext = 'VDKBNLDSNBNLOTSDQRDBTQHSX'
for i in range(26):
    if i <= 26:
        shift = i
        decrypted_text = decrypt_caesar_cipher(ciphertext, shift)
        print(decrypted_text)
        i += 1
