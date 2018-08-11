from gensim import similarities
from gensim.corpora import Dictionary
from gensim.models import LdaMulticore

from newlda.preprocess import Preprocessor

# config file here?


class SearchEngine(object):
    def __init__(self,
                 lda_model=None,
                 dictionary=None,
                 lda_path='lda_model',
                 dictionary_path='dictionary',
                 max_workers=4):
        self.lda_model = lda_model if lda_model is not None else LdaMulticore.load(lda_path)
        self.dictionary = dictionary if dictionary is not None else Dictionary.load(dictionary_path)
        self.preprocessor = Preprocessor(max_workers=max_workers)
        self.index = None
        self.urls = None

    def infer(self, document) :
        text = self.preprocessor.preprocess_doc(document)
        bow = self.dictionary.doc2bow(text)
        return self.lda_model[bow]

    def infer_all(self, docs_with_urls):
        preproc_docs_with_urls = self.preprocessor.process_docs_with_urls(docs_with_urls)
        bags_of_words = [(url, self.dictionary.doc2bow(doc)) for url, doc in preproc_docs_with_urls]
        return [(url, self.lda_model[bow]) for url, bow in bags_of_words]

    def dummy_index(self, docs_with_urls):
        urls, doc_bows = zip(*self.infer_all(docs_with_urls))
        self.urls = urls
        self.index = similarities.SparseMatrixSimilarity(doc_bows, num_features=400)
        self.index.save('/people/plgwciro/z4/z4/data/index')
        # self.save('urls', self.urls)

    # def save(self, filename, what):
    #     with open(filename, 'wb') as f:
    #         pickle.dump(what, f)
    #
    # def load(self, filename):
    #     with open(filename, 'rb') as f:
    #         return pickle.load(f)

    def load_index(self, path='index', urls='urls'):
        self.index = similarities.SparseMatrixSimilarity.load(path)
        # self.urls = self.load(urls)

    def dummy_search(self, query):
        inferred = self.index[self.infer(query)]
        ss = sorted(enumerate(inferred), key=lambda item: -item[1])
        return [(self.urls[i], sim) for i, sim in ss]

    def partial_search(self, query):
        infer_query = self.infer(query)
        inferred = self.index[infer_query]
        inferred = sorted(enumerate(inferred), key=lambda item: -item[1])[:500]

        query_len = len(query.split(" "))
        return {self.urls[i]: self.adjust(query_len, sim) for i, sim in inferred}

    def adjust(self, query_length, similarity):
        return similarity / ((5 - query_length) ** 2) if query_length < 4 else similarity