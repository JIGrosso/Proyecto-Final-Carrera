import json
from nltk.tokenize import sent_tokenize

print("Reading Text...")
with open('./outputs/preprocessed_splitted_input.json', encoding='utf8') as json_file:
    inputs = json.load(json_file)

outputs = []
for text_id in inputs:
    for p in inputs[text_id]:
        outputs = sent_tokenize(p)
        if len(outputs) > 1:
            print(outputs)




