import json
# from nltk.tokenize.punkt import PunktWordTokenizer, PunktSentenceTokenizer
from nltk.tokenize import TreebankWordTokenizer


# sentence_tokenizer = PunktSentenceTokenizer()
word_tokenizer = TreebankWordTokenizer()


def preprecess(file_path):
    neg_items = []
    pos_items = []

    with open(file_path, 'r', encoding='utf-8') as f_in:
        data = json.load(f_in)
        for item in data:
            print(item['id'])
            spans = [' '.join(word_tokenizer.tokenize(s.strip())) for s in item['spans']]
            sen = ' '.join(spans)
            if not sen:
                continue
            if float(item['sentiment score']) > 0:
                pos_items.append(sen.strip())
            else:
                neg_items.append(sen.strip())

    with open(file_path + '.neg', 'w', encoding='utf-8') as f_out:
        for line in neg_items:
            print(line, file=f_out)

    with open(file_path + '.pos', 'w', encoding='utf-8') as f_out:
        for line in pos_items:
            print(line, file=f_out)


if __name__ == "__main__":
    file_path = '../semeval-2017-task-5-subtask-1/Microblog_Trainingdata.json'
    preprecess(file_path)
