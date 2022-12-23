from sklearn.linear_model import LogisticRegression

import collections
import random
import re
import numpy as np
import pandas as pd
import copy
from joblib import load
import math
from statistics import mean
import time

from spellchecker import SpellChecker
from nltk.stem.porter import PorterStemmer
from conllu import parse
import ufal.udpipe

import streamlit as st

import os

#############---------------------------- Математические операции ----------------------------###############


def safe_divide(numerator, denominator):
    try:
        index = numerator / denominator
    except ZeroDivisionError:
        return 0


def division(list1, list2):
    try:
        return len(list1) / len(list2)
    except ZeroDivisionError:
        return 0


def corrected_division(list1, list2):
    try:
        return len(list1) / math.sqrt(2 * len(list2))
    except ZeroDivisionError:
        return 0


def root_division(list1, list2):
    try:
        return len(list1) / math.sqrt(len(list2))
    except ZeroDivisionError:
        return 0


def squared_division(list1, list2):
    try:
        return len(list1) ** 2 / len(list2)
    except ZeroDivisionError:
        return 0


def log_division(list1, list2):
    try:
        return math.log(len(list1)) / math.log(len(list2))
    except ZeroDivisionError:
        return 0


def uber(list1, list2):
    try:
        return math.log(len(list1)) ** 2 / math.log(len(set(list2)) / len(list1))
    except ZeroDivisionError:
        return 0


def levenshtein(seq1, seq2):
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = np.zeros((size_x, size_y))
    for x in range(size_x):
        matrix[x, 0] = x
    for y in range(size_y):
        matrix[0, y] = y

    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x-1] == seq2[y-1]:
                matrix[x, y] = min(
                    matrix[x-1, y] + 1,
                    matrix[x-1, y-1],
                    matrix[x, y-1] + 1
                )
            else:
                matrix[x, y] = min(
                    matrix[x-1, y] + 1,
                    matrix[x-1, y-1] + 1,
                    matrix[x, y-1] + 1
                )
    return (matrix[size_x - 1, size_y - 1])


#############---------------------------- Spellchecker ----------------------------###############
freq_r_path = os.path.abspath('russian.txt')
spell = SpellChecker(language='ru')
spell.word_frequency.load_text_file(freq_r_path)

#############---------------------------- Porter Stemmer ----------------------------###############

porter_stemmer = PorterStemmer()

#############---------------------------- Пути до нужных файлов ----------------------------###############

fr_5000 = os.path.abspath('fr_5000.txt')
with open(fr_5000, encoding='utf-8') as file:
    five_th_freq = file.read()
five_th_freq = [w[0] for w in re.findall('([А-ЯЁа-яё]+(-[А-ЯЁа-яё]+)*)', five_th_freq)]

verbs_fr_5000 = os.path.abspath('verbs_fr_5000.txt')
with open(verbs_fr_5000, encoding='utf-8') as file:
    verbs_five_th_freq = file.read()
verbs_five_th_freq = [w[0] for w in re.findall('([А-ЯЁа-яё]+(-[А-ЯЁа-яё]+)*)', verbs_five_th_freq)]

new_vocab_a1 = os.path.abspath('new_vocab_a1.txt')
with open(new_vocab_a1, encoding='utf-8') as file:
    a1_vocab = file.read()
a1_vocab = [w[0] for w in re.findall('([А-ЯЁа-яё]+(-[А-ЯЁа-яё]+)*)', a1_vocab)]

new_vocab_a2_short = os.path.abspath('new_vocab_a2_short.txt')
with open(new_vocab_a2_short, encoding='utf-8') as file:
    a2_vocab = file.read()
a2_vocab = [w[0] for w in re.findall('([А-ЯЁа-яё]+(-[А-ЯЁа-яё]+)*)', a2_vocab)]

new_vocab_b1_short = os.path.abspath('new_vocab_b1_short.txt')
with open(new_vocab_b1_short, encoding='utf-8') as file:
    b1_vocab = file.read()
b1_vocab = [w[0] for w in re.findall('([А-ЯЁа-яё]+(-[А-ЯЁа-яё]+)*)', b1_vocab)]

new_vocab_b2_short = os.path.abspath('new_vocab_b2_short.txt')
with open(new_vocab_b2_short, encoding='utf-8') as file:
    b2_vocab = file.read()
b2_vocab = [w[0] for w in re.findall('([А-ЯЁа-яё]+(-[А-ЯЁа-яё]+)*)', b2_vocab)]

new_vocab_c1_short = os.path.abspath('new_vocab_c1_short.txt')
with open(new_vocab_c1_short, encoding='utf-8') as file:
    c1_vocab = file.read()
c1_vocab = [w[0] for w in re.findall('([А-ЯЁа-яё]+(-[А-ЯЁа-яё]+)*)', c1_vocab)]

#############---------------------------- Части речи ----------------------------###############

OPEN_CLASS = ['NOUN', 'VERB', 'ADV', 'ADJ', 'PROPN']

#############---------------------------- Модель ----------------------------###############

class Model:
    def __init__(self, path):
        """Load given model."""
        self.model = ufal.udpipe.Model.load(path)
        if not self.model:
            raise Exception("Cannot load UDPipe model from file '%s'" % path)

    def tokenize(self, text):
        """Tokenize the text and return list of ufal.udpipe.Sentence-s."""
        tokenizer = self.model.newTokenizer(self.model.DEFAULT)
        if not tokenizer:
            raise Exception("The model does not have a tokenizer")
        return self._read(text, tokenizer)

    def read(self, text, in_format):
        """Load text in the given format (conllu|horizontal|vertical) and return list of ufal.udpipe.Sentence-s."""
        input_format = ufal.udpipe.InputFormat.newInputFormat(in_format)
        if not input_format:
            raise Exception("Cannot create input format '%s'" % in_format)
        return self._read(text, input_format)

    def _read(self, text, input_format):
        input_format.setText(text)
        error = ufal.udpipe.ProcessingError()
        sentences = []

        sentence = ufal.udpipe.Sentence()
        while input_format.nextSentence(sentence, error):
            sentences.append(sentence)
            sentence = ufal.udpipe.Sentence()
        if error.occurred():
            raise Exception(error.message)

        return sentences

    def tag(self, sentence):
        """Tag the given ufal.udpipe.Sentence (inplace)."""
        self.model.tag(sentence, self.model.DEFAULT)

    def parse(self, sentence):
        """Parse the given ufal.udpipe.Sentence (inplace)."""
        self.model.parse(sentence, self.model.DEFAULT)

    def write(self, sentences, out_format):
        """Write given ufal.udpipe.Sentence-s in the required format (conllu|horizontal|vertical)."""

        output_format = ufal.udpipe.OutputFormat.newOutputFormat(out_format)
        output = ''
        for sentence in sentences:
            output += output_format.writeSentence(sentence)
        output += output_format.finishDocument()

        return output

model_p = os.path.abspath('russian-syntagrus-ud-2.5-191206.udpipe')
model = Model(model_p) 

#############---------------------------- Парсер ----------------------------###############

class ParserUDpipe:
    """Parses text using UDpipe."""

    def __init__(self):
        self.text = ''
        self.conllu = ''
        self.lemmas = []
        self.tokens = []
        self.verb_lemmas = []
        self.noun_lemmas = []
        self.adj_lemmas = []
        self.adv_lemmas = []
        self.open_class_lemmas = []
        self.infinitive_tokens = []
        self.pres_sg_tokens = []
        self.verb_tokens = []
        self.aux_forms = []
        self.pres_pl_tokens = []
        self.parts = []
        self.pasts = []
        self.finite_tokens = []
        self.sentences = []
        self.relations = []
        self.pos_tags = []
        self.finite_forms = []

        self.pos_lemma = {}

    def text2conllu(self, text, model):
        self.text = text
        sentences = model.tokenize(self.text)
        for s in sentences:
            model.tag(s)
            model.parse(s)
        self.conllu = model.write(sentences, 'conllu')

    def get_info(self):
        self.lemmas = []
        self.tokens = []
        self.verb_lemmas = []
        self.noun_lemmas = []
        self.adj_lemmas = []
        self.adv_lemmas = []
        self.open_class_lemmas = []
        self.infinitive_tokens = []
        self.pres_sg_tokens = []
        self.verb_tokens = []
        self.aux_forms = []
        self.pres_pl_tokens = []
        self.parts = []
        self.pasts = []
        self.finite_tokens = []
        self.sentences = []
        self.relations = []
        self.pos_tags = []
        self.finite_forms = []
        self.finite_forms = []
        self.pos_lemma = {}

        self.sentences = parse(self.conllu)
        for x in range(len(self.sentences)):
            self.pos_lemma[x] = [[], []]

        for i, sentence in enumerate(self.sentences):
            finite_forms_one = []
            finite_deps_one = []
            coord_one = []
            for token in sentence:

                lemma = token.get('lemma')
                form = token.get('form')
                relation = token.get('deprel')
                pos = token.get('upostag')
                feats = token.get('feats')
                head = token.get('head')
                self.pos_lemma[i][0].append(pos)
                self.pos_lemma[i][1].append(lemma)

                self.relations.append(relation)
                self.lemmas.append(lemma)
                self.tokens.append(form)
                self.pos_tags.append(pos)
                if not feats:
                    feats = {}

                if pos == 'VERB':
                    self.verb_lemmas.append(lemma)
                    self.verb_tokens.append(form)
                    if feats.get('VerbForm', '') == 'Fin':
                        self.finite_tokens.append(form)
                if pos == 'NOUN':
                    self.noun_lemmas.append(lemma)
                if pos == 'ADJ':
                    self.adj_lemmas.append(lemma)
                if pos == 'ADV':
                    self.adv_lemmas.append(lemma)
                if pos == 'AUX':
                    self.aux_forms.append(form)
                if pos in OPEN_CLASS:
                    self.open_class_lemmas.append(lemma)
                if feats.get('VerbForm', '') == 'Inf':
                    self.infinitive_tokens.append(form)
                if feats.get('Person', '') == '3' and \
                        feats.get('Tense', '') == 'Pres' and\
                        feats.get('Mood', '') == 'Ind' and\
                        feats.get('VerbForm', '') == 'Fin':
                    if feats.get('Number', '') == 'Sing':
                        self.pres_sg_tokens.append(form)
                    if feats.get('Number', '') == 'Plur':
                        self.pres_pl_tokens.append(form)
                if feats.get('Tense', '') == 'Past' and feats.get('VerbForm', '') == 'Part':
                    self.parts.append(form)
                if feats.get('Mood', '') == 'Ind' and \
                        feats.get('Person', '') == '3' and\
                        feats.get('Tense', '') == 'Past' and\
                        feats.get('VerbForm', '') == 'Fin':
                    self.pasts.append(form)
                if feats.get('VerbForm', '') == 'Fin':
                    finite_forms_one.append(form)

            self.finite_forms.append(finite_forms_one)

parser = ParserUDpipe()

#############---------------------------- Достаем фичи ----------------------------###############

class GetFeatures:
    """Returns values of complexity criteria."""

    def __init__(self, model):
        self.model = model
        self.text = ''
        self.lemmas = []
        self.tokens = []
        self.verb_lemmas = []
        self.noun_lemmas = []
        self.adj_lemmas = []
        self.adv_lemmas = []
        self.open_class_lemmas = []
        self.infinitive_tokens = []
        self.pres_sg_tokens = []
        self.verb_tokens = []
        self.aux_forms = []
        self.pres_pl_tokens = []
        self.parts = []
        self.pasts = []
        self.finite_tokens = []
        self.sentences = []
        self.relations = []
        self.pos_tags = []
        self.finite_forms = []
        self.pos_lemma = {}

    def get_info(self, text):
        self.text = text
        parser.text2conllu(self.text, self.model)
        parser.get_info()
        self.lemmas = parser.lemmas
        self.tokens = parser.tokens
        self.verb_lemmas = parser.verb_lemmas
        self.noun_lemmas = parser.noun_lemmas
        self.adj_lemmas = parser.adj_lemmas
        self.adv_lemmas = parser.adv_lemmas
        self.open_class_lemmas = parser.open_class_lemmas
        self.infinitive_tokens = parser.infinitive_tokens
        self.pres_sg_tokens = parser.pres_sg_tokens
        self.verb_tokens = parser.verb_tokens
        self.aux_forms = parser.aux_forms
        self.pres_pl_tokens = parser.pres_pl_tokens
        self.parts = parser.parts
        self.pasts = parser.pasts
        self.finite_tokens = parser.finite_tokens
        self.sentences = parser.sentences
        self.relations = parser.relations
        self.pos_tags = parser.pos_tags
        self.finite_forms = parser.finite_forms
        self.pos_lemma = parser.pos_lemma
        
# Lexical diversity measures

    def NDW(self):
        """
        number of lemmas
        """
        return len(set(self.lemmas))
    
    def TTR(self):
        """
        number of lemmas/number of tokens
        """
        lemmas = set(self.lemmas)
        tokens = self.tokens
        TTR = division(lemmas, tokens)
        CTTR = corrected_division(lemmas, tokens)
        RTTR = root_division(lemmas, tokens)
        LogTTR = log_division(lemmas, tokens)
        Uber = uber(lemmas, tokens)
        return TTR, CTTR, RTTR, LogTTR, Uber

    
    def LV(self):
        """
        number of lexical lemmas/number of lexical tokens
        """
        lex_lemmas = set(self.lemmas)
        lex_tokens = self.tokens
        return len(lex_lemmas) / len(lex_tokens)
    
    def VV(self):
        """
        VVI: number of verb lemmas/number of verb tokens
        VVII: number of verb lemmas/number of lexical tokens
        """
        verb_lemmas = set(self.verb_lemmas)
        verb_tokens = self.verb_lemmas
        lex_tokens = self.open_class_lemmas
        VVI = division(verb_lemmas, verb_tokens)
        SVVI = squared_division(verb_lemmas, verb_tokens)
        CVVI = corrected_division(verb_lemmas, verb_tokens)
        VVII = division(verb_lemmas, lex_tokens)
        return VVI, SVVI, CVVI, VVII
    
    def NV(self):
        """
        number of noun lemmas/number of lexical tokens
        """
        noun_lemmas = set(self.noun_lemmas)
        lex_tokens = self.tokens
        return division(noun_lemmas, lex_tokens)
    
    def AdjV(self):
        """
        number of adjective lemmas/number of lexical tokens
        """
        adj_lemmas = set(self.adj_lemmas)
        lex_tokens = self.open_class_lemmas
        return division(adj_lemmas, lex_tokens)
    
    def AdvV(self):
        """
        number of adverb lemmas/number of lexical tokens
        """
        adv_lemmas = set(self.adv_lemmas)
        lex_tokens = self.open_class_lemmas
        return division(adv_lemmas, lex_tokens)
    
    def ModV(self):
        return self.AdjV() + self.AdvV()

# Lexical density

    def density(self):
        """
        number of lexical tokens/number of tokens
        """
        return division(self.open_class_lemmas, self.lemmas)
    
# Lexical sophistication measures

    def LS(self):
        """
        number of sophisticated lexical tokens/number of lexical tokens
        """
        soph_lex_lemmas = [i for i in self.open_class_lemmas if i not in five_th_freq]
        return division(soph_lex_lemmas, self.open_class_lemmas)
    
    def LFP(self):
        """
        Lexical Frequency Profile is the proportion of tokens:
        first - 1000 most frequent words
        second list - the second 1000
        none - list of those that are not in these lists
        """
        first = [i for i in self.lemmas if i in five_th_freq[0:1000]]
        second = [i for i in self.lemmas if i in five_th_freq[1001:2000]]
        third = [i for i in self.lemmas if i in five_th_freq[2001:]]
        first_procent = division(first, self.lemmas)
        second_procent = division(second, self.lemmas)
        third_procent = division(third, self.lemmas)
        none = 1 - (first_procent + second_procent + third_procent)
        return first_procent, second_procent, third_procent, none

    def VS(self):
        """
        number of sophisticated verb lemmas/number of verb tokens
        """
        soph_verbs = [i for i in self.verb_lemmas if i not in verbs_five_th_freq]
        VSI = division(soph_verbs, self.verb_lemmas)
        VSII = corrected_division(soph_verbs, self.verb_lemmas)
        VSIII = squared_division(soph_verbs, self.verb_lemmas)
        return VSI, VSII, VSIII
    
    def choose(self, n, k):
        """
        Calculates binomial coefficients
        """
        if 0 <= k <= n:
            ntok = 1
            ktok = 1
            for t in range(1, min(k, n - k) + 1):
                ntok *= n
                ktok *= t
                n -= 1
                return ntok // ktok
            else:
                return 0

    def num_uniques(self, l):
        counter = collections.Counter(l)
        return list(counter.values()).count(1)

    def freq_finite_forms(self):
        """
        frequency of tensed(finite) forms
        """
        return division(self.finite_tokens, self.verb_tokens)

    def freq_aux(self):
        """
        frequency of modals(auxilaries)
        """
        return division(self.aux_forms, self.verb_tokens)

    def subfinder(self, mylist, pattern):
        matches = []
        for i in range(len(mylist)):
            if mylist[i] == pattern[0] and mylist[i:i + len(pattern)] == pattern:
                matches.append(pattern)
        return matches

    def num_dict_2_levels(self, d, prefix):
        num_all = 0
        result = {}
        for group in d:
            num_group = 0
            for subgroup in d[group]:
                num_subgroup = 0
                name_subgroup = list(subgroup.keys())[0]
                for word in list(subgroup.values())[0]:
                    num = len(re.findall(word.lower(), self.text.lower()))
                    num_all += num
                    num_subgroup += num
                    num_group += num
                    result[prefix + name_subgroup + "(" + word + ")"] = num
                result[prefix + name_subgroup] = num_subgroup
            result[prefix + group] = num_group
        result[prefix + 'all'] = num_all
        return result

    def order_head(self, sentence):
        ids = []
        heads = []
        for i, token in enumerate(sentence, start=1):
            heads.append(token.get('head'))
            ids.append(i)
        return (list(zip(ids, heads)))

    def find_root(self, order_head_lst):
        for every_order_head in order_head_lst:
            if every_order_head[1] == 0:
                root = every_order_head
        return root

    def root_children(self, sentence):
        order_head_lst = self.order_head(sentence)
        root = self.find_root(order_head_lst)
        chains = []
        for every_order_head in order_head_lst:
            if every_order_head[1] == root[0]:
                chains.append([root[0], every_order_head[0]])
        return chains, order_head_lst

    def chains_heads(self, chains, order_head_lst):
        length_chains = len(chains)
        i = 0
        for chain in chains:
            if i < length_chains:
                heads = []
                if 'stop' not in chain:
                    for order_head in order_head_lst:
                        if chain[-1] == order_head[1]:
                            heads.append(order_head[0])
                    if heads == [] and 'stop' not in chain:
                        chain.append('stop')
                    else:
                        ind_head = 0
                        for head in heads:
                            new_chain = copy.copy(chain)[:-1]
                            if ind_head == 0:
                                chain.append(head)
                                ind_head += 1
                            else:
                                new_chain.append(head)
                                chains.append(new_chain)
            i += 1
        while all(item[-1] == 'stop' for item in chains) is False:
            self.chains_heads(chains, order_head_lst)
        return chains

    def count_depth_for_one_sent(self, sentence):
        chains, order_head_lst = self.root_children(sentence)
        chains = self.chains_heads(chains, order_head_lst)
        depths = []
        for chain in chains:
            depths.append(len(chain) - 2)
        if depths:
            return max(depths)
        else:
            return 0

    def count_depths(self):
        max_depths = []
        for sentence in self.sentences:
            depth = self.count_depth_for_one_sent(sentence)
            max_depths.append(depth)
        return max_depths

    def av_depth(self):
        max_depths = self.count_depths()
        return np.mean(max_depths)

    def max_depth(self):
        max_depths = self.count_depths()
        return np.max(max_depths)

    def min_depth(self):
        max_depths = self.count_depths()
        return np.min(max_depths)

    def count_dep_sent(self):
        dict_dep_rel = collections.Counter(self.relations)
        acl = dict_dep_rel.get('acl', 0)
        rel_cl = dict_dep_rel.get('acl:relcl', 0)
        advcl = dict_dep_rel.get('advcl', 0)
        return acl, rel_cl, advcl

    def count_sent(self):
        return len(self.sentences)

    def count_tokens(self, punct=True):
        if punct:
            return len(self.pos_tags)
        else:
            return len([x for x in self.pos_tags if x != 'PUNCT'])

    def tokens_before_root(self):
        length = []
        for sentence in self.sentences:
            for i, token in enumerate(sentence):
                rel_type = token.get('deprel')
                if rel_type == 'root':
                    break
            length.append(i)
        return mean(length)

    def mean_len_sent(self, punct=True):
        length = []
        for sentence in self.sentences:
            i = 0
            for token in sentence:
                if punct:
                    i += 1
                else:
                    pos = token.get('upostag')
                    if pos != 'PUNCT':
                        i += 1
            length.append(i)
        return mean(length)

    def count_adj_noun(self):
        num_adj_noun = 0
        for sentence in self.sentences:
            for token in sentence:
                pos = token.get('upostag')
                head = token.get('head')
                if head:
                    pos_head = sentence[head - 1].get('upostag')
                    if pos == 'ADJ' and pos_head == 'NOUN':
                        num_adj_noun += 1
        return num_adj_noun

    def count_part_noun(self):
        num_part_noun = 0
        for sentence in self.sentences:
            for token in sentence:
                pos = token.get('feats')
                if not pos:
                    pos = {}
                pos = pos.get('VerbForm')
                head = token.get('head')
                if head:
                    pos_head = sentence[head - 1].get('upostag')
                    if pos == 'Part' and pos_head == 'NOUN':
                        num_part_noun += 1
        return num_part_noun

    def count_noun_inf(self):
        num_inf_noun = 0
        for sentence in self.sentences:
            for token in sentence:
                pos = token.get('feats')
                if not pos:
                    pos = {}
                pos = pos.get('VerbForm')
                head = token.get('head')
                if head:
                    pos_head = sentence[head - 1].get('upostag')
                    if pos == 'Inf' and pos_head == 'NOUN':
                        num_inf_noun += 1
        return num_inf_noun

    def simularity_mean(self):
        mean_pos_sim, mean_lemma_sim = [], []
        for id_1, sentence_1 in enumerate(self.pos_lemma):
            sum_dist_pos, sum_dist_lemma = [], []
            for id_2, sentence_2 in enumerate(self.pos_lemma):
                if id_1 != id_2 and id_1 < id_2:
                    dist_pos = levenshtein(self.pos_lemma[id_1][0], self.pos_lemma[id_2][0])
                    dist_lemma = levenshtein(self.pos_lemma[id_1][1], self.pos_lemma[id_2][1])
                    sum_dist_pos.append(dist_pos)
                    sum_dist_lemma.append(dist_lemma)
            if sum_dist_pos:
                mean_pos_sim.append(mean(sum_dist_pos))
                mean_lemma_sim.append(mean(sum_dist_lemma))
        try:
            return mean(mean_pos_sim), mean(mean_lemma_sim)
        except ValueError:
            return 0, 0

    def simularity_neibour(self):
        mean_pos_sim_nei, mean_lemma_sim_nei = [], []
        for i, sentence_1 in enumerate(self.pos_lemma):
            if i + 1 < len(self.pos_lemma):
                mean_pos_sim_nei.append(levenshtein(self.pos_lemma[i][0], self.pos_lemma[i + 1][0]))
                mean_lemma_sim_nei.append(levenshtein(self.pos_lemma[i][1], self.pos_lemma[i + 1][1]))
        try:
            return mean(mean_pos_sim_nei), mean(mean_lemma_sim_nei)
        except ValueError:
            return 0, 0
        
    def number_of_misspelled(self):
        misspelled = spell.unknown(self.tokens)
        return misspelled
    
    def vocab_level(self):
        """
        number of used vocabulary by levels 
        """
        lvl_a_1 = [i for i in self.lemmas if i in a1_vocab]
        lvl_a_2 = [i for i in self.lemmas if i in a2_vocab]
        lvl_b_1 = [i for i in self.lemmas if i in b1_vocab]
        lvl_b_2 = [i for i in self.lemmas if i in b2_vocab]
        lvl_c_1 = [i for i in self.lemmas if i in c1_vocab]
        a1_procent = len(lvl_a_1)
        a2_procent = len(lvl_a_2)
        b1_procent = len(lvl_b_1)
        b2_procent = len(lvl_b_2)
        c1_procent = len(lvl_c_1)
        none = len(self.lemmas) - (a1_procent + a2_procent + b1_procent + b2_procent + c1_procent)
        
        return a1_procent, a2_procent, b1_procent, b2_procent, c1_procent, none

gf = GetFeatures(model)

#############---------------------------- ОБЩИЙ СБОР!!!11!1 ----------------------------###############

def main(text):
    gf.get_info(text)
    result = [0] * 41
    result[0] = gf.av_depth()
    result[1] = gf.max_depth()
    result[2] = gf.min_depth()
    result[3], result[4], result[5] = gf.count_dep_sent()
    result[5] = gf.count_sent()
    result[6] = gf.count_tokens()
    result[7] = gf.tokens_before_root()
    result[8] = gf.mean_len_sent()
    result[9] = gf.count_adj_noun()
    result[10] = gf.count_part_noun()
    result[11] = gf.count_noun_inf()
    result[12], result[13] = gf.simularity_neibour()
    result[14], result[15] = gf.simularity_mean()
    result[16] = gf.density()
    result[17] = gf.LS()
    result[18], result[19], result[20] = gf.VS()
    result[21], result[22], result[23], result[24] = gf.LFP()
    result[25] = gf.NDW()
    result[26], result[27], result[28], result[29], result[30] = gf.TTR()
    result[31] = gf.LV()
    result[32], result[33], result[34], result[35] = gf.VV()
    result[36] = gf.NV()
    result[37] = gf.AdjV()
    result[38] = gf.AdvV()
    result[39] = gf.ModV()
    result[40] = gf.freq_finite_forms()
    return result

def erros_number(text):
    gf.get_info(text)
    result = {}
    result['num_misspelled_tokens'] = gf.number_of_misspelled()
    return result

def lex_level(text):
    gf.get_info(text)
    result = {}
    result['A1_vocabulary'], result['A2_vocabulary'], result['B1_vocabulary'], result['B2_vocabulary'], result['C1_vocabulary'], result['Not in lists'] = gf.vocab_level()
    return result

#############---------------------------- обучение модели ----------------------------###############

essays = os.path.abspath('anonymous_placement_test2022.xlsx')
essays = pd.read_excel(essays) 

essays = essays.dropna()

def proc(x):
    if x[-2].isdigit(): 
        return int(x[-2])
    else:
        return int(x[-1])
#     return x[-2]
    

levels = essays['level_num'].map(proc)

vectors = essays['essay'].map(main)
# vectors = np.array([v[0] for v in vectors])
vectors = list(vectors)

labels_3 = [1 if x>= 3 else 0 for x in levels]
labels_4 = [1 if x>= 4 else 0 for x in levels]
labels_5 = [1 if x>= 5 else 0 for x in levels]
labels_6 = [1 if x>= 6 else 0 for x in levels]

clf_3 = LogisticRegression(random_state=0, max_iter=1000000).fit(vectors, labels_3)
results_3 = clf_3.predict_proba(vectors)
clf_4 = LogisticRegression(random_state=0, max_iter=1000000).fit(vectors, labels_4)
results_4 = clf_4.predict_proba(vectors)
clf_5 = LogisticRegression(random_state=0, max_iter=1000000).fit(vectors, labels_5)
results_5 = clf_5.predict_proba(vectors)
clf_6 = LogisticRegression(random_state=0, max_iter=1000000).fit(vectors, labels_6)
results_6 = clf_6.predict_proba(vectors)

#вставьте текст между кавычками
############---------------------------- ввод текста ----------------------------###############
#тут будет код для той библиотеки
st.sidebar.header(':red[О проекте]')
st.sidebar.markdown('Инструмент разработан в рамках научно-исследовательского семинара за 2021-2022 год. Автор -  :red[Белова Татьяна], студентка магистратуры по направлению "Компьютерная лингвистика" НИУ ВШЭ \n\n Научный руководитель - :red[Клышинский Эдуард Станиславович] \n\n Проект посвящен исследованию способов автоматической оценки эссе и прогнозирования уровня владения русским языком как иностранным. Мы разработали систему оценки текста на русском языке, основанную на сложности текста, которая включает морфологические, лексические и синтаксические особенности \n\n Мы используем регрессионую модель, основанную на модели UDPipe Russian-SynTagRus 2.5, параметры сложности из статьи INSPECTOR Tool, словарный минимум TORFL и средство проверки орфографии для оценки эссе студентов')
st.sidebar.markdown(':red[[Репозиторий проекта](https://github.com/tatiana-belova/russian_tool)]')

st.title('Определение уровня владения русским языком на основе анализа письменной речи')

st.subheader('Введите ваш текст в поле ниже')
text = st.text_area('label', placeholder = 'Ваш текст', label_visibility="collapsed")

def has_lat(text):
    return bool(re.search('([A-Za-z]+(-[A-Za-z]+)*)', text))

def has_rus(text):
    return bool(re.search('([А-ЯЁа-яё]+(-[А-ЯЁа-яё]+)*)', text))

if st.button('Проверить мой текст'):
    if len(text) == 0:
        st.error('Пожалуйста, введите текст в поле выше')

    elif len(text) > 0 and has_lat(text) == True:
        st.error('Пожалуйста, введите текст на русском языке')
            
    elif len(text) > 0 and has_rus(text) == True:
        vector_text = np.asarray(main(text))
        mist_text = erros_number(text)
        vocab_lvl = lex_level(text)
    
        results_3 = clf_3.predict_proba(vector_text.reshape(1, -1))
        results_4 = clf_4.predict_proba(vector_text.reshape(1, -1))
        results_5 = clf_5.predict_proba(vector_text.reshape(1, -1))
        results_6 = clf_6.predict_proba(vector_text.reshape(1, -1))
    
        result_3 = sum(results_3.tolist(), [])
        result_4 = sum(results_4.tolist(), [])
        result_5 = sum(results_5.tolist(), [])
        result_6 = sum(results_6.tolist(), [])
    
        if result_3[1] > 0.9 and result_4[1] > 0.9 and result_5[1] > 0.9 and result_6[1] > 0.72:
            st.subheader('Уровень владения: **:green[Третий уровень (ТРКИ-III)/C1]**')
            st.markdown(':green[Третий уровень (ТРКИ-III)/C1] присваивается тем, кто  владеет русским языком на высоком уровне, сравнимым с уровнем носителя языка')
        elif result_3[1] > 0.9 and result_4[1] > 0.9 and result_5[1] > 0.9 and result_6[1] < 0.72:
            st.subheader('Уровень владения: **:violet[Второй уровень (ТРКИ-II)/B2]**')
            st.markdown(':violet[Второй уровень (ТРКИ-II)/B2] присваивается тем, кто имеет достаточно высокий уровень владения русским языком, который позволяет кандидату удовлетворить коммунникативные потребности во всех сферах общения')
        elif result_3[1] > 0.9 and result_4[1] > 0.9 and result_5[1] < 0.9 and result_6[1] < 0.72:
            st.subheader('Уровень владения: **:blue[Первый уровень (ТРКИ-I)/B1]**')
            st.markdown(':blue[Первый уровень (ТРКИ-I)/B1] присваивается тем, кто имеет средний уровень владения русским языком, который позволяет кандидату удовлетворить основные коммуникативные потребности в бытовой, учебной и профессиональной сферах общения в соответствии с государственным стандартом русского языка как иностранного языка')
        elif result_3[1] > 0.9 and result_4[1] < 0.9 and result_5[1] < 0.9 and result_6[1] < 0.72:
            st.subheader('Уровень владения: **:red[Базовый уровень (ТБУ)/А2]**')
            st.markdown(':red[Базовый уровень (ТБУ)/А2] присваивается тем, кто владеет начальным уровнем знаний русского языка, достаточным для основных коммуникативных потребностей в ограниченном числе ситуаций бытовой и культурной сфер общения')
        elif result_3[1] < 0.9 and result_4[1] < 0.9 and result_5[1] < 0.9 and result_6[1] < 0.72:
            st.subheader('Уровень владения: **:orange[Элементарный уровень (ТЭУ)/А1]**')
            st.markdown(':orange[Элементарный уровень (ТЭУ)/А1] присваивается тем, кто владеет минимальным уровнем знаний русского языка, достаточным для ограниченного числа ситуаций в повседневном общении')
             
    else:
        pass

