import functools

from nltk.corpus import wordnet as wn
import spacy

nlp = spacy.load('en_core_web_sm')

class AnswerType:
    def __init__(self,
                 seeds={'person.n.01': ("who", "what"),
                        'location.n.01': ("where", "what"),
                        'time_period.n.01': ("when", "in what"),
                        'battle.n.01': ("where", "at what")},
                 det_default='what',
                 qw_default='what',
                 special_cases={}):
        self.special_cases = special_cases

        self.det_lookup = {}
        self.qw_lookup = {}
        
        self.qw_default = qw_default
        self.det_default = det_default

        for synset in seeds:
            question_word, question_determiner = seeds[synset]
            for hyponym in wn.synset(synset).closure(lambda x: x.hyponyms()):
                for lemma in hyponym.lemmas():
                    self.det_lookup[lemma.name().lower()] = question_determiner
                    self.qw_lookup[lemma.name().lower()] = question_word

        self.plural = {}

    @functools.lru_cache(maxsize=5000)
    def is_plural(self, lexical_answer_type):
        analysis = nlp(lexical_answer_type)

        return any(x.tag_ in {"NNS", "NNPS"})

    def question_word(self, mention):
        return self.qw_lookup.get(mention, self.qw_default)

    def determiner(self, mention):
        return self.det_lookup.get(mention, self.det_default)


if __name__ == "__main__":
    at = AnswerType()
    print(" ".join("%s:%s" % (x, at.qw_lookup[x]) for x in at.det_lookup if x.startswith("cou")))
    for ii in ["country"]:
        print(ii, at.question_word(ii), at.determiner(ii))
