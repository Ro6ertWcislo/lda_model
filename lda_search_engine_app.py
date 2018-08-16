import sys

from model.util.file_parser import parse_dir_json
from search_engine.lda.search_engine import SearchEngine
from search_engine.lda.logger.logger_config import init_logger
from configuration.lda.config import LdaConfig

if __name__ == '__main__':
    init_logger()

    config = LdaConfig(sys.argv[1],'lda_search_engine').get_current_config()

    docs = parse_dir_json(config['data_path'])

    searchEngine = SearchEngine(config['topics'])
    searchEngine.load_model(config['model_path'], config['dict_path'])
    searchEngine.dummy_index(docs)

    searchEngine.save_index(config['index_path'], config['url_path'])
    searchEngine.infer("what a wonderfull day")
