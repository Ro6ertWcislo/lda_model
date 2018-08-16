import sys

from model.lda.lda_model import LDA
from model.util.file_parser import parse_dir_json
from search_engine.lda.search_engine import SearchEngine
from configuration.lda.config import LdaConfig

if __name__ == '__main__':
    config = LdaConfig(sys.argv[1]).get_current_config()

    docs = parse_dir_json(config['data_path'])

    lda = LDA.with_url_handling()

    lda.train(docs)
    lda.save_model(config['model_path'])
    lda.save_dictionary('dict_path')

    searchEngine = SearchEngine(lda_model=lda.model, dictionary=lda.dictionary)
    searchEngine.dummy_index(docs)
