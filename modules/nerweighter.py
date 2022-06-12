from natasha import (
    Segmenter,
    MorphVocab,

    NewsEmbedding,
    NewsMorphTagger,
    NewsSyntaxParser,
    NewsNERTagger,

    PER,
    NamesExtractor,

    Doc
)
import json
from string import punctuation
import numpy as np


def softmax(x):

    y = np.exp(x - np.max(x))
    f_x = y / np.sum(np.exp(x))
    return f_x

def get_nerwindow_comparison(fake, potorigs, idf_dict=None, window=5, koefspans=True):

    segmenter = Segmenter()
    morph_vocab = MorphVocab()

    emb = NewsEmbedding()
    morph_tagger = NewsMorphTagger(emb)
    syntax_parser = NewsSyntaxParser(emb)
    ner_tagger = NewsNERTagger(emb)

    names_extractor = NamesExtractor(morph_vocab)

    texts = [fake] + potorigs
    nerdict = []
    lsums = {i: 0 for i in range(1, len(texts))}
    sumidf= {i: 0 for i in range(1, len(texts))}
    for w in range(2, window+1):
        for ind, text in enumerate(texts):
            nerdict.append({})
            doc = Doc(text)
            doc.segment(segmenter)
            doc.tag_morph(morph_tagger)
            for token in doc.tokens:
                token.lemmatize(morph_vocab)
            doc.parse_syntax(syntax_parser)
            doc.tag_ner(ner_tagger)
            for span in doc.spans:
                span.normalize(morph_vocab)
            for token in doc.tokens:
                text = text.replace(token.text, token.lemma)
            text = text.replace('\n', ' ')
            for p in punctuation + '«»':
                text = text.replace(p, '')
            text = np.array([x for x in text.split() if len(x) > 0])
            spans = set()
            if idf_dict and ind != 0:
                sumidf[ind] = 0
            for span in doc.spans:
                span = span.normal.lower()
                for p in punctuation + '«»':
                    span = span.replace(p, '')
                if idf_dict and ind != 0 and span in idf_dict:
                    sumidf[ind] += idf_dict[span]
                spans.add(span)
            for span in spans:
                nerdict[-1][span] = set()
                idxs = np.where(text == span)[0]
                for idx in idxs:
                    [nerdict[-1][span].add(x)
                     for x in text[idx - w:idx] if len(x) > 2]
                    [nerdict[-1][span].add(x)
                     for x in text[idx+1:idx+w] if len(x) > 2]
        for i in range(1, len(texts)):
            lsum = 0
            for span in nerdict[i].keys():
                if len(nerdict[i][span]) > 0:
                    if span in nerdict[0]:
                        value = len(set.intersection(nerdict[0][span], nerdict[i][span]))/len(nerdict[i][span])
                        if koefspans and value > 0:
                            impwords_local = len(set.intersection(nerdict[i][span], spans))
                            if impwords_local > 0:
                                value += len(set.intersection(set.intersection(nerdict[0][span], nerdict[i][span]), spans))/impwords_local
                        if idf_dict and span in idf_dict:
                            value *= idf_dict[span]/sumidf[i]
                        lsum += value
            lsum /= len(nerdict[i])
            lsums[i] += lsum
    for p in lsums:
        lsums[p] = 1.0 - lsums[p]/(window-1)
    #maxi = max(np.array([x for x in list(lsums.values()) if x != 1]))
    # softmaxed = softmax(np.array([x for x in list(lsums.values()) if x != 0]))
    # shift = 0
    # for i,p in enumerate(lsums):
    #     if lsums[p] != 0:
    #         lsums[p] = softmaxed[i - shift]
    #     else:
    #         shift += 1
    # maxi = max(np.array([x for x in list(lsums.values()) if x != 1]))
    lower = 0.975
    for p in lsums:
        if lsums[p] != 1:
            lsums[p] = (lsums[p] - lower)/(1-lower)
            while lsums[p] > 1:
                lower += 0.005
                lsums[p] = (lsums[p] - lower)/(1-lower)
            while lsums[p] < 0:
                lower -= 0.005
                lsums[p] = (lsums[p] - lower)/(1-lower)

        if potorigs[p-1] == fake:
            lsums[p] = 0.0
    return lsums

if __name__ == '__main__':
    from matcher import Matcher
    matcher = Matcher('../resources/news.csv')
    fake = """Москва стала самым популярным направлением для поездок на июньские праздники, сообщила Наталья Сергунина, заместитель Мэра Москвы. В пятерку также вошли Санкт-Петербург, Республика Татарстан, Нижегородская и Владимирская области. По данным туристического сервисаRusspass, у тех, кто собирается посетить столицу с 11 по 13 июня, высоким спросом пользуются интерактивные и тематические экскурсии для отдыха с семьей и авторские маршруты. По ее словам, в топ-5 маршрутов по Москве в выходные вошли«Архитектурная прогулка по ВДНХ»,«Знакомство с Царицыно»,«Прогулка по “Зарядью”»,«Выходной в русском Версале»и«Москва в деталях: прогулка по Парку Горького». Второе место по популярности занял Санкт-Петербург. Наряду с поездками в Северную столицу путешественники отдали предпочтение изучению окрестностей города и близлежащих направлений. """
    #print(matcher.match(["Москва заняла первое место среди европейских городов в рейтинге инноваций, помогающих в борьбе с COVID-19"]).title)
    texts = list(matcher.match(["Москва возглавила рейтинг популярных туристических направлений России на июньские праздники"]).text)
    with open('../resources/mos_idf.json') as fin:
        gdict = json.load(fin)
    #print(texts[0])
    print(get_nerwindow_comparison(fake, texts, idf_dict=gdict, window=10))