from spacy.tokenizer import Tokenizer
from spacy.lang.en import English
from spacy.lang.fr import French
from spacy.lang.de import German

nlp_en = English()
nlp_fr = French()
nlp_de = German()

def tokenize_en(text):
    return [t.text for t in nlp_en(text)]

def tokenize_fr(text):
    return [t.text for t in nlp_fr(text)]

def tokenize_de(text):
    return [t.text for t in nlp_de(text)]

def get_tokenizer(lang):
    if lang=='.en':
        return tokenize_en
    elif lang=='.fr':
        return tokenize_fr
    elif lang=='.de':
        return tokenize_de
    else:
        raise ValueError('Unknown language')