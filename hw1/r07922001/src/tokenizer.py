from nltk.corpus import wordnet
from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer


class Tokenizer:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
    def get_wordnet_pos(self, treebank_tag):
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            return None
    def lemmatize_sentence(self, sentence):
        res = []
        for word, pos in pos_tag(word_tokenize(sentence)):
            wordnet_pos = self.get_wordnet_pos(pos) or wordnet.NOUN
            res.append(self.lemmatizer.lemmatize(word, pos=wordnet_pos))
        return res
