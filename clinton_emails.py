from snorkel.models import Span, Label
from sqlalchemy.orm.exc import NoResultFound


def add_candidate_label(session, key, cls, person1, organization, value):
    try:
        person1 = session.query(Span).filter(Span.stable_id == person1).one()
        organization = session.query(Span).filter(Span.stable_id == organization).one()
    except NoResultFound as e:
        if int(value) == -1:
            ### Due to variations in the NER output of CoreNLP, some of the included annotations for
            ### false candidates might cover slightly different text spans when run on some systems,
            ### so we must skip them.
            return
        else:
            raise e
    candidate = session.query(cls).filter(cls.person1 == person1).filter(cls.organization == organization).first()
    if candidate is None:
        candidate = session.query(cls).filter(cls.person1 == organization).filter(cls.organization == person1).one()

    label = session.query(Label).filter(Label.candidate == candidate).one_or_none()
    if label is None:
        label = Label(candidate=candidate, key=key, value=value)
        session.add(label)
    else:
        label.value = int(value)
    session.commit()



def number_of_people(sentence):
    active_sequence = False
    count = 0
    for tag in sentence.ner_tags:
        if tag == 'PERSON' and not active_sequence:
            active_sequence = True
            count += 1
        elif tag != 'PERSON' and active_sequence:
            active_sequence = False
    return count
    

def sent_has_org(sentence):
    for tag in sentence.ner_tags:
        if tag=='ORGANIZATION':
            return True
    return False
        

import re
from snorkel.lf_helpers import get_left_tokens, get_right_tokens, get_between_tokens, get_text_between



titles={'MP','UNDER SECRETARY','Envoy,Representative','Assistant','Special Assistant','diplomat','official','government official','AMBASSADOR','Chancellor','Sen','Senator','Congresswoman','Congressman','Chief of Staff','mayor','Chairman','Attorney General','General',' Gen','Vice President','VP','President','dictator','Defense Secretary','Secretary of State','Secretary General','Gov.','Governer','Speaker','House Speaker','Democrat','Republican','foreign minister','Prime Minister','ambassador','amb','Founder','Co-Founder','Author','chief executive','CEO','head  of','editor','reporter','publisher','anchor','adviser','Chairman','chairwoman','chair','Rep.','columnist','militant','director','deputy director','Executive Director','professor','Navy SEAL','talk show host','premier'}

betweentags= {'of the','nominated','appointed','served','elected','to the','he is','she is','head of'}

def LF_too_far_apart(c):
    return -1 if len(get_between_tokens(c)) > 10 else 0

def LF_third_wheel(c):
    return -1 if 'PERSON' in get_between_tokens(c, attrib='ner_tags', case_sensitive=True) else 0
    
#def LF_nospace(c):
#    return 1 if [] == get_between_tokens(c)  else 0

#def LF_betweentags(c):
#    return 1 if len(betweentags.intersection(set(get_between_tokens(c)))) >0  and len(get_between_tokens(c)) < 5 else 0

def LF_title_right_before_or_after(c):
    if len(titles.intersection(set(c[0].parent.words))) == 0:
        if len(titles.intersection(set(get_left_tokens(c[0],window=5, attrib='words')))) > 0 or len(titles.intersection(set(get_right_tokens(c[0],window=5, attrib='words')))) > 0:
            return 1
        else:
            return 0
    else:
        return 0

def LF_betweentokens(c):
    if len(get_between_tokens(c)) < 3:
        return 1
    else:
        return 0

def LF_betweentags(c):
    if len(get_between_tokens(c)) < 10 and  len(betweentags.intersection(set(get_right_tokens(c[0],window=8,attrib='words')))) == 0:
        return 1
    else:
        return 0


def LF_title_in_sentence(c):
    if len(titles.intersection(set(c[0].parent.words))) == 0 :
        return 0
    else:
        return -1



LFs = [LF_too_far_apart, LF_third_wheel, LF_title_in_sentence, LF_betweentags,LF_title_right_before_or_after,LF_betweentokens]


from snorkel import SnorkelSession
session = SnorkelSession()
import os

from snorkel.parser import TSVDocParser
doc_parser = TSVDocParser(path="data/clinton_train.tsv")

from snorkel.parser import SentenceParser

sent_parser = SentenceParser()
from snorkel.parser import CorpusParser

cp = CorpusParser(doc_parser, sent_parser)
%time corpus = cp.parse_corpus(session, "Emails Training")
session.add(corpus)
session.commit()


for name, path in [('Emails Development', 'data/clinton_dev.tsv'),
                   ('Emails Test', 'data/clinton_test.tsv')]:
    doc_parser.path=path
    %time corpus = cp.parse_corpus(session, name)
    session.commit()

from snorkel.models import Corpus

corpus = session.query(Corpus).filter(Corpus.name == 'Emails Training').one()

sentences = set()
for document in corpus:
    for sentence in document.sentences:
        if number_of_people(sentence) < 5:
            sentences.add(sentence)





from snorkel.models import candidate_subclass


Title = candidate_subclass('Person_Org', ['person1','Org'])

from snorkel.candidates import Ngrams

ngrams = Ngrams(n_max=3)

ngrams_org = Ngrams(n_max=8)

from snorkel.matchers import PersonMatcher

from snorkel.matchers import OrganizationMatcher
from snorkel.matchers import DictionaryMatch
longest_match_only = True

person_matcher = PersonMatcher(longest_match_only=longest_match_only)

org_matcher = OrganizationMatcher(longest_match_only=longest_match_only)


title_matcher = DictionaryMatch(d=titles, ignore_case=True, 
                                longest_match_only=longest_match_only)

from snorkel.candidates import CandidateExtractor

ce = CandidateExtractor(Title, [ngrams,ngrams_org], [person_matcher,org_matcher],
                        symmetric_relations=False, nested_relations=False, self_relations=False)
						
%time c = ce.extract(sentences, 'Emails Training Candidates', session)
print "Number of candidates:", len(c)

session.add(c)
session.commit()
from snorkel.models import Corpus
for corpus_name in ['Emails Development', 'Emails Test']:
    corpus = session.query(Corpus).filter(Corpus.name == corpus_name).one()
    sentences = set()
    for document in corpus:
        for sentence in document.sentences:
            if number_of_people(sentence) < 5:
                sentences.add(sentence)
    
    %time c = ce.extract(sentences, corpus_name + ' Candidates', session)
    session.add(c)
session.commit()

from snorkel.models import CandidateSet

train = session.query(CandidateSet).filter(CandidateSet.name == 'Emails Training Candidates').one()
dev = session.query(CandidateSet).filter(CandidateSet.name == 'Emails Development Candidates').one()

from snorkel.annotations import FeatureManager

feature_manager = FeatureManager()

%time F_train = feature_manager.create(session, c, 'Train Features')


#To load existing use ..
#%time F_train = feature_manager.load(session, train, 'Train Features')						
						
from snorkel.annotations import LabelManager

label_manager = LabelManager()

%time L_train = label_manager.create(session, c, 'LF Labels', f=LFs)
L_train

from snorkel.learning import NaiveBayes

gen_model = NaiveBayes()
gen_model.train(L_train, n_iter=1000, rate=1e-5)


gen_model.save(session, 'Generative Params')
train_marginals = gen_model.marginals(L_train)
gen_model.w

from snorkel.learning import LogReg
from snorkel.learning_utils import RandomSearch, ListParameter, RangeParameter

iter_param = ListParameter('n_iter', [250, 500, 1000, 2000])
rate_param = RangeParameter('rate', 1e-4, 1e-2, step=0.75, log_base=10)
reg_param  = RangeParameter('mu', 1e-8, 1e-2, step=1, log_base=10)

disc_model = LogReg()

%time F_dev = feature_manager.update(session, dev, 'Train Features', False)

searcher = RandomSearch(disc_model, F_train, train_marginals, 10, iter_param, rate_param, reg_param)