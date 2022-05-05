import spacy
import neuralcoref

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

    def chunk_from_mention(self, mention_tokens, lexical_answer_type):
        all_ancestors = set()

        left_edge = float('inf')
        right_edge = 0 
        for ii in mention_tokens:
            left_edge = min(ii.i, left_edge)
            right_edge = max(ii.i, right_edge)
            all_ancestors = all_ancestors.union(set(ii.ancestors))
            print(left_edge, right_edge, all_ancestors)

        for subtree in [list(x.subtree) for x in all_ancestors if x.pos_ == 'VERB']:
            left_subtree = " ".join(x.text for x in subtree if x.i < left_edge)
            right_subtree = " ".join(x.text for x in subtree if x.i > right_edge)
            yield " ".join([left_subtree, lexical_answer_type, right_subtree])
            

            
    def relative_inside_last_mention(self, mention_tokens, lexical_answer_type):
        all_ancestors = set()
        for ii in mention_tokens:
            all_ancestors = all_ancestors.union(set(ii.ancestors))

        right_edge = 0
        left_edge = float('inf')
            
        for verb in [x for x in all_ancestors if x.pos_ == "VERB"]:
            right_edge = max(right_edge, verb.right_edge.i)
            left_edge = min(left_edge, verb.left_edge.i)

        return " ".join(x.text for x in mention_tokens if (x.i >= left_edge and x.i <= right_edge))
        
    def generate_chunks(self, lexical_answer_type):
        nominals = [list(x) for x in question.answer_nominal_mentions()]
        pronominals = list(question.answer_pronominal_mentions())

        chunks = set()

        for ii in nominals + pronominals:
            chunks.add(question.chunk_from_mention(ii, lexical_answer_type))

        for ii in chunks:
            yield ii

        for ii in self.relative_inside_last_mention(nominals[-1], lexical_answer_type):
            yield ii

        
if __name__ == "__main__":

    questions = []
    for qid, page, text, lat in [(0, "Alps", "Three watercolors depict this location in dark, red, and blue tones at different times of day. Zurich is found in this highest mountain range of Europe.  Another painting set at this location features a rock inscribed with KAROLUS MAGNUS IMP. It is the home to St. Bernhard's pass.  Hannibal cross them with elephants.  In response to that work, Paul Delaroche painted an exhausted mule trudging through this location. In a more grandiose painting set here, a title figure in a billowing cloak points upward astride a rearing (*) horse. A massive storm cloud arcs over a yellow-orange sun in a J.M.W. Turner painting of an army crossing this location. For 10 points, give this location, depicted in a Jacques-Louis David portrait of Napoleon passing through this European mountain range.", "which mountains"),
                                (1, "Dido", "According to legend, she fled Phoenicia for North Africa after her brother Pygmalion killed her husband Sychaeus. In North Mrica she founded the city of Carthage and reigned as its queen. FTP, identify this queen, who committed suicide after Aeneas left her.", "which ruler")]:
        question = Question(qid, page, text)
        questions.append(question)
        
        nominals = [list(x) for x in question.answer_nominal_mentions()]

        for ii in nominals:
            for ii in question.chunk_from_mention(ii, lat):
                print("NOM CHUNK", ii)

        pronominals = list(question.answer_pronominal_mentions())

        for ii in pronominals:
            print("PRO:", ii)
            for ii in question.chunk_from_mention(ii, lat):
                print("PRO CHUNK", ii)

        print("LAST RELATIVE", question.relative_inside_last_mention(nominals[-1], lat))
