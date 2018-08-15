import logging

import sys

from model.lda.lda_model import LDA
from model.util.file_parser import parse_dir_json
from search_engine.lda.search_engine import SearchEngine
from model.lda.config.config import LdaConfig

def init_logger():
    logger = logging.getLogger('lda_model')
    logger.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    fh = logging.FileHandler('lda_model.log')
    fh.setLevel(logging.ERROR)

    # create console handler with a higher log level
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setLevel(logging.DEBUG)

    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)


if __name__ == '__main__':
    init_logger()
    config = LdaConfig(sys.argv[1]).get_current_config()

    docs = parse_dir_json(config['data_path'])

    lda = LDA.with_url_handling()

    model, dictionary = lda.train(docs)
    model.save(config['model_path'])
    dictionary.save('dict_path')

    searchEngine = SearchEngine(lda_model=model, dictionary=dictionary)
    searchEngine.dummy_index(docs)
