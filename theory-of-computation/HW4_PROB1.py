import re

f = open("Encoded_Image_by_noise.txt")

# function to filter the junk characters
def filtering(str):
    junk = '!', '\"', '#', '$', '%', '&', '\'', '(', ')', '*', ',', '-', '.', ':', ';', '<', '>', '?', '@', '[', ']', '\\', '^', '_', '`', '{', '}', '|', '~'
    pattern = '|'.join(re.escape(char) for char in junk)    # escape function that helps escape special characters
    filtered_string = re.sub(pattern, '', str)              # substitute function that replaces the junk characters with nothing
    return filtered_string

# reading the file and store all the text in a string
unfiltered_string = ''
for i in f.read():
    unfiltered_string = unfiltered_string + i

# calling the filtering function and printing the result
a = filtering(unfiltered_string)
print(a)

f.close()
