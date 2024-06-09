import re

def regex_split(text):
    return re.findall(r'\b\w+\b', text)

text = "This is a more complex example, using regex for tokenization."
tokens = regex_split(text)
print(tokens)
