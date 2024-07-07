import re
from icecream import ic

def find_first_char_after_pattern(text):
    # Compile a regular expression to match the pattern like ```json\n, ```yml\n, etc.
    regex = re.compile(r'```.*?\n')
    
    # Search for the pattern in the text
    match = regex.search(text)
    
    if match:
        # The position of the first character after the matched pattern
        return match.end()
    else:
        # If the pattern is not found, return -1 or any other indication of failure
        return -1

if __name__ == "__main__":
    with open("original_ec2.json", "r") as file:
        text = file.read()
        ic(text)
        pos = find_first_char_after_pattern(text)
        ic(text[pos:])