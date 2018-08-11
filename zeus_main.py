from newlda.lda_model import LDA
from newlda.file_parser import parse_dir_json
from newlda.search_engine import SearchEngine

docs = parse_dir_json("lda_model/data/")

lda = LDA.with_url_handling()

model, dictionary = lda.train(docs)
model.save("/people/plgwciro/lda_model/model/lda_jsn")
dictionary.save("/people/plgwciro/lda_model/model/dict_jsn")

searchEngine = SearchEngine(lda_model=model, dictionary=dictionary)
searchEngine.dummy_index(docs)
