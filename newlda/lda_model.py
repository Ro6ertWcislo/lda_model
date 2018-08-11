
from concurrent.futures import ProcessPoolExecutor

from gensim.corpora import Dictionary
from gensim.models import LdaModel, LdaMulticore

from newlda.preprocess import Preprocessor, WithUrlPreprocessor


class LDA(object):
    def __init__(self, max_workers= 4,
                 save= False,
                 num_topics = 30,
                 passes = 20,
                 lda_filename= "lda_model",
                 dict_filename= "dictionary",
                 preprocessor=None):
        self.lda_filename = lda_filename
        self.dict_filename = dict_filename
        self.passes = passes
        self.num_topics = num_topics
        self.save = save
        self.max_workers = max_workers
        self.preprocessor = preprocessor if preprocessor is not None else Preprocessor(max_workers=max_workers)

    def train(self, doc_list):
        preprocessed_docs = self.preprocessor.process_docs(doc_list)

        dictionary = Dictionary(preprocessed_docs)

        corpus = self.build_corpus(preprocessed_docs, dictionary)

        model = LdaMulticore(corpus,
                             num_topics=self.num_topics,
                             id2word=dictionary,
                             workers=self.max_workers,
                             passes=self.passes)

        # x = models.LdaModel(corpus, num_topics=30, id2word=dictionary, passes=20)
        if self.save:
            model.save(self.lda_filename)
            dictionary.save(self.dict_filename)

        return model, dictionary

    def build_corpus(self, doc_list, dictionary):  # list(corpora.Dictionary)
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            return list(executor.map(dictionary.doc2bow, doc_list))

    @staticmethod
    def with_url_handling(max_workers: int = 4,
                          save: bool = False,
                          num_topics: int = 30,
                          passes: int = 20,
                          lda_filename: str = "lda_model",
                          dict_filename: str = "dictionary"):
        return LDA(max_workers=max_workers,
                   save=save,
                   num_topics=num_topics,
                   passes=passes,
                   lda_filename=lda_filename,
                   dict_filename=dict_filename,
                   preprocessor=WithUrlPreprocessor(max_workers=max_workers))
