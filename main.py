from modules.model import NNDefaker
from modules.matcher import Matcher
from modules.nerweighter import get_nerwindow_comparison
import sys
import json

WEIGHTS = [0.2, 0.8]

if __name__ == '__main__':
    matcher = Matcher('./resources/news.csv')
    with open('./resources/mos_idf.json') as fin:
        idf_dict = json.load(fin)
    matched = matcher.match([sys.argv[2]])
    nerprobs = get_nerwindow_comparison(sys.argv[3], list(matched.text))
    matched['probs'] = list(nerprobs.values())
    # defaker = NNDefaker(sys.argv[1])
    # nnprobs = list(defaker.infer_text([sys.argv[3]] + list(matched.text)))[1:]
    # matched['nnprobs'] = nnprobs
    # matched['probs'] = WEIGHTS[0] * matched['probs'] + WEIGHTS[1] * matched['nnrprobs']
    print(list(zip(matched.title, matched.probs)))
    
