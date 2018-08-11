import logging

from newlda.lda_model import LDA
from newlda.file_parser import parse_dir_json
from newlda.search_engine import SearchEngine


def init_logger():
    # create logger with 'spam_application'
    logger = logging.getLogger('lda_model')
    logger.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    fh = logging.FileHandler('lda_model.log')
    fh.setLevel(logging.ERROR)
    fh.setLevel(logging.WARNING)

    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setLevel(logging.DEBUG)
    ch.setLevel(logging.WARNING)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)


if __name__ == '__main__':
    init_logger()
    docs = parse_dir_json("lda_model/data/")

    lda = LDA.with_url_handling()

    model, dictionary = lda.train(docs)
    model.save("/people/plgwciro/lda_model/model/lda_jsn")
    dictionary.save("/people/plgwciro/lda_model/model/dict_jsn")

    searchEngine = SearchEngine(lda_model=model, dictionary=dictionary)
    searchEngine.dummy_index(docs)
