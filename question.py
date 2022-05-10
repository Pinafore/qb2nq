import spacy
import neuralcoref
import logging

nlp = spacy.load('en_core_web_sm')
neuralcoref.add_to_pipe(nlp)


class Question:

    def __init__(self, qanta_id, page, text,
                     exclude_pronouns={"her", "his", "their", "its"},
                     triggers={'this', 'these', 'what'},
                     directive_verbs={"name", "identify", "give"}):
        self.qid = qanta_id
        self.page = page
        self.raw_text = text
        self.exclude_pronouns = exclude_pronouns
        self.directive_verbs = directive_verbs
        self.triggers = triggers

        self.analysis = nlp(text)

    def answer_nominal_mentions(self):
        for this_idx in [idx for idx, token in enumerate(self.analysis) if token.text.lower() in self.triggers]:
            for parent in self.analysis[this_idx].ancestors:
                if self.analysis[parent.i].pos_ == "NOUN":
                    yield self.analysis[parent.i].subtree
                    break

    def answer_pronominal_mentions(self, min_mentions=3):
        clusters = self.analysis._.coref_clusters

        if clusters:
            largest_cluster_size = max(len(x.mentions) for x in clusters)
            if largest_cluster_size < min_mentions:
                return

            for cluster in clusters:
                if len(cluster.mentions) == largest_cluster_size:
                    for mention in cluster.mentions:
                        if mention[0].text not in self.exclude_pronouns:
                            yield mention

    def sentences(self):
        for sent in self.analysis.sents:
            yield [x.text for x in sent]
                            
    def is_posessive(self, mention_tokens):
        if mention_tokens[-1].text == "'s" or mention_tokens[-1].text.lower() in {"his", "her", "their"}:
            return True
        else:
            return False
                            
    def chunk_from_mention(self, mention_tokens, substitution, posessive_substitution):
        all_ancestors = set()

        if self.is_posessive(mention_tokens):
            substitution = posessive_substitution

        left_edge = float('inf')
        right_edge = 0 
        for ii in mention_tokens:
            left_edge = min(ii.i, left_edge)
            right_edge = max(ii.i, right_edge)
            all_ancestors = all_ancestors.union(set(ii.ancestors))


        for subtree in [list(x.subtree) for x in all_ancestors if x.pos_ == 'VERB']:
            left_subtree = " ".join(x.text for x in subtree if x.i < left_edge)
            right_subtree = " ".join(x.text for x in subtree if x.i > right_edge)
            logging.debug("[Normal] Left: %s Mention: (%s -> %s) Right: %s" % (left_subtree, " ".join(x.text for x in mention_tokens), substitution, right_subtree))
            yield " ".join([left_subtree, substitution, right_subtree])
            

            
    def relative_inside_last_mention(self, mention_tokens, lexical_answer_type):
        all_ancestors = set()
        for ii in mention_tokens:
            all_ancestors = all_ancestors.union(set(ii.ancestors))

        right_edge = 0
        left_edge = float('inf')
            
        for verb in [x for x in all_ancestors if x.pos_ == "VERB"]:
            right_edge = max(right_edge, verb.right_edge.i)
            left_edge = min(left_edge, verb.left_edge.i)

        logging.debug(("[LAST] Left", left_edge, "Right", right_edge))
        return [x.text for x in mention_tokens if (x.i >= left_edge and x.i <= right_edge)]
        
    def generate_chunks(self, lexical_answer_type, question_word):
        nominals = [list(x) for x in self.answer_nominal_mentions()]
        pronominals = list(self.answer_pronominal_mentions())

        chunks = set()

        # TODO (jbg): If making a posessive becomes more complicated, turn this into a function
        for ii in nominals + pronominals:
            chunks.add(self.chunk_from_mention(ii, lexical_answer_type, "%s 's" % lexical_answer_type))
            chunks.add(self.chunk_from_mention(ii, question_word, "whose"))

        for ii in chunks:
            yield ii

        yield self.relative_inside_last_mention(nominals[-1], lexical_answer_type)
        yield self.relative_inside_last_mention(nominals[-1], question_word)
        
if __name__ == "__main__":
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
    questions = []
    for qid, page, text, lat, qw in [(0, "Alps", "Three watercolors depict this location in dark, red, and blue tones at different times of day. Zurich is found in this highest mountain range of Europe.  Another painting set at this location features a rock inscribed with KAROLUS MAGNUS IMP. It is the home to St. Bernhard's pass.  Hannibal cross them with elephants.  In response to that work, Paul Delaroche painted an exhausted mule trudging through this location. In a more grandiose painting set here, a title figure in a billowing cloak points upward astride a rearing (*) horse. A massive storm cloud arcs over a yellow-orange sun in a J.M.W. Turner painting of an army crossing this location. For 10 points, give this location, depicted in a Jacques-Louis David portrait of Napoleon passing through this European mountain range.", "which mountains", "where"),
                                (1, "Dido", "According to legend, she fled Phoenicia for North Africa after her brother Pygmalion killed her husband Sychaeus. In North Mrica she founded the city of Carthage and reigned as its queen. FTP, identify this queen, who committed suicide after Aeneas left her.", "which ruler", "who"),
                                         (2, "Zwingli", "This man criticized his country's involvement in foreign wars in an allegorical fable called The Ox. He demanded compensation for the family of Jacob Kaiser and forced another group to end its alliance with Austria in an armistice that he negotiated to end a war in which no battles occurred. This man executed Felix Mantz, a founder of a group known as his nation's Brethren. That group later became the Anabaptists, whom this man attacked with tracts like Tricks of the Catabaptists. He was succeeded by Heinrich Bullinger after his death during the Second Kappel War. His denial of the real presence of Christ in the Eucharist caused his split with Martin Luther at the Marburg Colloquy. For 10 points, name this Swiss Protestant reformer who preached at Zürich.", "what man", "who"),
                                         (3, "Zjord", "The Solarljod claims that this figure's eldest daughter is called Radvör and that his youngest is called Kreppvör. Loki insulted this god by saying that the daughters of Hymir relieved themselves in his mouth. This figure's father-in-law once captured Loki as a giant eagle, forcing Loki to lure Idunn into Thrymheim. This figure was exchanged with Hoenir to end the Aesir-Vanir War. The wife of this god was once made to laugh by the efforts of Loki and a female goat. He separated from his wife because she could not get used to living in Nóatún due to the screeching of the gulls after she selected this god for his beautiful feet. For 10 points, name this father of Freyr and Freya and husband of Skadi, the Norse god of wealth and seafaring.", "which god", "who")]:
        question = Question(qid, page, text)
        questions.append(question)
        
        for ii in question.generate_chunks(lat, qw):
            print(" ".join(ii))
