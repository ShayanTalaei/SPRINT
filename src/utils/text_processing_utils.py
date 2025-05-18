import re
import json
import networkx as nx

alphabets= "([A-Za-z])"
prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|Prof|Capt|Cpt|Lt|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov|edu|me)"
digits = "([0-9])"
multiple_dots = r'\.{2,}'
multiple_dots_followed_by_comma = r'\.{2,},'

def split_into_sentences(text: str) -> list[str]:
    """
    Split the text into sentences.

    If the text contains substrings "<prd>" or "<stop>", they would lead 
    to incorrect splitting because they are used as markers for splitting.

    :param text: text to be split into sentences
    :type text: str

    :return: list of sentences
    :rtype: list[str]
    """
    text = " " + text + "  "
    text = re.sub(r'(\n\d+)\. ', r'\1<prd> ', text) # For numbered lists with space after dot
    text = re.sub(r'(\n+)', lambda match: match.group(0) + "<stop>", text)
    text = re.sub(prefixes,"\\1<prd>",text)
    text = re.sub(websites,"<prd>\\1",text)
    text = re.sub(digits + "[.]" + digits,"\\1<prd>\\2",text)
    text = re.sub(multiple_dots_followed_by_comma, lambda match: "<prd>" * len(match.group(0)) + ",", text)
    text = re.sub(multiple_dots, lambda match: "<prd>" * len(match.group(0)) + "<stop>", text)
    if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
    text = re.sub("\s" + alphabets + "[.] "," \\1<prd> ",text)
    text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>",text)
    text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
    text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
    text = re.sub(" " + alphabets + "[.]"," \\1<prd>",text)
    if "”" in text: text = text.replace(".”","”.")
    if "\"" in text: text = text.replace(".\"","\".")
    if "!" in text: text = text.replace("!\"","\"!")
    if "?" in text: text = text.replace("?\"","\"?")
    text = re.sub(r'\.(?!\n)', '.<stop>', text)
    text = re.sub(r'\?(?!\n)', '?<stop>', text)
    text = re.sub(r'!(?!\n)', '!<stop>', text)
    text = text.replace("<prd>",".")
    sentences = text.split("<stop>")
    
    sentences = [s.strip(" ") for s in sentences]
    if sentences and not sentences[-1]: sentences = sentences[:-1]
    return sentences

def extract_trailing_newlines(s):
    match = re.search(r'(\n+)$', s)
    if match:
        trailing_newlines = match.group(1)
        s = s[:match.start()]
        return trailing_newlines, s
    return '', s

def read_file_as_string(file_path):
    # open the file in read mode and read its content
    with open(file_path, 'r') as file:
        file_content = file.read()
    return file_content

def read_json_as_dag(file_path):
    with open(file_path, "r") as file:
        parent_dictionary = json.load(file)
    parent_dictionary = {int(i):[int(k) for k in j] for i, j in parent_dictionary.items()}
    
    # Create a directed graph in networkx
    G = nx.DiGraph()
    for node, parents in parent_dictionary.items():
        G.add_node(node)
        for parent in parents:
            G.add_edge(parent, node)
    return G