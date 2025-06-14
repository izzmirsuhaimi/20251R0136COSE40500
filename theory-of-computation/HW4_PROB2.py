from bs4 import BeautifulSoup
import re

leet_dict = {                                       # this dictionary is organized in a certain way
    '/\/\\': 'M',
    '/\/': 'N',
    '/\\': 'A',
    '|3': 'B',
    '(_,)': 'Q',
    '(_)': 'U',
    '\/\/': 'W',
    '\/': 'V',
    '><': 'X',
    '-\-': 'Z'
}

f = open("ILLEET_LEETs.html", "r")                  # open the html file
html_file = f.read()                                # read the contents of the html file
soup = BeautifulSoup(html_file, 'html.parser')      # parse the html file using BeautifulSoup module
parsed_html = soup.prettify()                       # use the prettify function in BeautifulSoup module to separate the tags

def convert_from_leet(text, leet_dict):
    pattern = '|'.join(map(re.escape, leet_dict.keys()))    # map function to execute re.escape for each dict keys
    #print(pattern)
    def replace(match):
        return leet_dict[match.group(0)]
    return re.sub(pattern, replace, text)

converted_text = convert_from_leet(parsed_html, leet_dict)
print(converted_text)
f.close()

f = open("ILLEET_LEETs.html", "w")                  # open the html file again
f.write(converted_text)                             # overwrite the new content
f.close()