class Token:

    def __init__(self, word, lemma, pos):
        self.word = word
        self.lemma = lemma
        self.pos = pos

    def num_chars(self):
        return len(self.word)
    
    def __repr__(self):
        return f"Token(word={self.word}, lemma={self.lemma}, pos={self.pos})"


class Sentence:

    def __init__(self, text, tokens=None):
        self.text = text
        if tokens is None:
            self.tokens = []
        else:
            self.tokens = tokens

    def add_token(self, token):
        self.tokens.append(token)
    
    def get_words(self):
        words = []
        for token in self.tokens:
            words.append(token.word)
        return words


    def get_lemmas(self):
        lemmas = []
        for token in self.tokens:
            lemmas.append(token.lemma)
        return lemmas
    
    def get_pos_tags(self):
        pos_tags = []
        for token in self.tokens:
            pos_tags.append(token.pos)
        return pos_tags
    
    def num_tokens(self):
        return len(self.tokens)

    def num_chars(self):
        # characters of the words + spaces between words
        if not self.tokens: # check that the sentence actually has tokens
            return 0
        return sum(t.num_chars() for t in self.tokens) + (len(self.tokens) - 1)

    def __repr__(self):
        return f"Sentence(text={self.text}, tokens={self.tokens})"


class Document:
    
    def __init__(self, path, doc_id, split, label, sentences=None, features=None):
        self.path = path            # path to the document's .conllu file
        self.doc_id =  doc_id       # document id
        self.split = split          # train/test
        self.label = label          # label used for classification (topic)
        if sentences is None:
            self.sentences = []
        else:
            self.sentences = sentences
        if features is None:
            self.features = {}
        else:
            self.features = features

    def add_sentence(self, sentence):
        self.sentences.append(sentence)

    def get_tokens(self):
        tokens = []
        for sentence in self.sentences:
            for token in sentence.tokens:
                tokens.append(token)
        return tokens
        # return [t for s in self.sentences for t in s.tokens]
    
    def get_words(self):
        words = []
        for sentence in self.sentences:
            for token in sentence.tokens:
                words.append(token.word)
        return words
    
    def get_lemmas(self):
        lemmas = []
        for sentence in self.sentences:
            for token in sentence.tokens:
                lemmas.append(token.lemma)
        return lemmas

    def get_pos_tags(self):
        pos_tags = []
        for sentence in self.sentences:
            for token in sentence.tokens:
                pos_tags.append(token.pos)
        return pos_tags

    def num_tokens(self):
        num_tokens = 0
        for sentence in self.sentences:
            num_tokens += sentence.num_tokens()
        return num_tokens
    
    def num_chars(self):
        num_chars = 0
        for sentence in self.sentences:
            num_chars += sentence.num_chars()
        return num_chars


    def num_sentences(self):
        return len(self.sentences)

    def load_sentences_from_conllu(self):

        with open(self.path, "r", encoding="utf8") as f:
            for line in f:
                line = line.rstrip("\n")

                # end of sentence
                if line == "":
                    if sentence.num_tokens() > 0:
                        self.add_sentence(sentence)
                    continue

                # CoNLL-U comments
                if line.startswith("# text = "):
                    text = line[len("# text = "):]
                    sentence = Sentence(text=text)
                    continue
                if line.startswith("#"):
                    continue

                cols = line.split("\t")
             
                tok_id = cols[0]
                # skip multiword tokens (1-2) 
                if "-" in tok_id:
                    continue

                word, lemma, pos = cols[1], cols[2], cols[3]
                token = Token(word=word, lemma=lemma, pos=pos)
                sentence.add_token(token)

        # last sentence in case the file doesn't end with a blank line
        if sentence.num_tokens() > 0:
            self.add_sentence(sentence)

    def __repr__(self):
        return f"Document(path={self.path}, doc_id={self.doc_id}, split={self.split}, label={self.label}, num_sentences={len(self.sentences)})"
