import re

f = open("Secret disk.txt", "r")
file = f.read()

def extract_numbers(file):
    phone_pattern = '\\b\d{3}-\d{4}-\d{4}\\b'
    card_pattern = '\\b\d{4}-\d{4}-\d{4}-\d{4}\\b'      # \\b to treat it as backslash, not as escape character

    phone_numbers = re.findall(phone_pattern, file)
    card_numbers = re.findall(card_pattern, file)

    return phone_numbers, card_numbers

phone_numbers, card_numbers = extract_numbers(file)

print("Phone number")
for i in phone_numbers:
    print(i)
print("\nCard number")
for j in card_numbers:
    print(j)

f.close()