import loader
import numpy as np
import _pickle as cPickle
import sys
import time
import torch

from model import BiLSTM
from utils import adjust_learning_rate
import utils


test_path = ".\\dataset\\eng.testa"
pre_embed_path = ".\\glove.6B.100d.txt"
model_path = ".\\BiLSTM_CRF.pth"
mapping_file = ".\\dataset\\map_data.map"



mapping = {}
with open(mapping_file, 'rb') as f:
    mapping = cPickle.load(f)

word_to_id = mapping['word_to_id']
tag_to_id = mapping['tag_to_id']
char_to_id = mapping['char_to_id']
word_embeds = mapping['word_embeds'] 

model = BiLSTM(
    voca_size=len(word_to_id),
    word_emb_dim=100,
    pre_word_emb=word_embeds,
    char_emb_dim=25,
    char_lstm_dim=25,
    char_to_ix=char_to_id,
    n_cap=4,
    cap_emb_dim=8,
    hidden_dim=200,
    tag_to_ix=tag_to_id
)


x = torch.load(model_path)
model.load_state_dict(x())

model.eval()

def test():
    test_sentences = loader.load_data(test_path, zeros=False)

    loader.update_tag_scheme(test_sentences, 'iob')

    test_data = loader.pepare_dataset(
        test_sentences, word_to_id, char_to_id, tag_to_id
    )

    print("%i sentences in test." % (len(test_data)))



    confusion_matrix = torch.zeros((len(tag_to_id) - 2, len(tag_to_id) - 2))
    for data in test_data:
        sentence_in = data['words']
        sentence_in = torch.tensor(sentence_in, dtype=torch.long)
        
        cap_in = data['caps']
        cap_in = torch.tensor(cap_in, dtype=torch.long)

        tags = data['tags']
        tags = torch.tensor(tags)

        chars2 = data['chars']
        chars2_sorted = sorted(chars2, key=lambda p: len(p), reverse=True)
        d = {}
        for i, ci in enumerate(chars2):
            for j, cj in enumerate(chars2_sorted):
                if ci==cj and not j in d and not i in d.values():
                    d[j] = i
                    continue
        chars2_length = [len(w) for w in chars2_sorted]
        char_maxl = max(chars2_length)
        chars2_mask = np.zeros((len(chars2_sorted), char_maxl), dtype='int')
        for i, c in enumerate(chars2_sorted):
            chars2_mask[i, :chars2_length[i]] = c
        chars2_mask = torch.tensor(chars2_mask, dtype=torch.long)

        val, out = model(sentence_in, chars2_mask, cap_in, chars2_length, d)
        predicted_id = out
        for (true_id, pred_id) in zip(tags, predicted_id):
            confusion_matrix[true_id, pred_id] += 1
        
    num_correct = 0
    for i in range(confusion_matrix.size(0)):
        num_correct += confusion_matrix[i][i]

    accurency = num_correct / np.sum(confusion_matrix.data.numpy())
    print(accurency.data)

def NER_for_sentence(sentence):
    sentence = utils.remove_numbers(sentence)
    sentence = utils.remove_punctua(sentence)
    sentence = utils.remove_whitespace(sentence)
    str_words = sentence.split()
    # print(str_words)

    data = loader.prepare_sentence(str_words, word_to_id, char_to_id, tag_to_id)
    sentence_in = data['words']
    sentence_in = torch.tensor(sentence_in, dtype=torch.long)
    
    cap_in = data['caps']
    cap_in = torch.tensor(cap_in, dtype=torch.long)

    chars2 = data['chars']
    chars2_sorted = sorted(chars2, key=lambda p: len(p), reverse=True)
    d = {}
    for i, ci in enumerate(chars2):
        for j, cj in enumerate(chars2_sorted):
            if ci==cj and not j in d and not i in d.values():
                d[j] = i
                continue
    chars2_length = [len(w) for w in chars2_sorted]
    char_maxl = max(chars2_length)
    chars2_mask = np.zeros((len(chars2_sorted), char_maxl), dtype='int')
    for i, c in enumerate(chars2_sorted):
        chars2_mask[i, :chars2_length[i]] = c
    chars2_mask = torch.tensor(chars2_mask, dtype=torch.long)

    val, out = model(sentence_in, chars2_mask, cap_in, chars2_length, d)
    predicted_id = out
    # print(out)
    id_to_tag = {v: i for i, v in tag_to_id.items()}
    tags = [id_to_tag[id.item()] for id in predicted_id]
    result = []
    for item in zip(str_words, tags):
        result.append(item)

    return result

test()

# tags = NER_for_sentence("Dr Chris Muldoon is a former Guide Dogs instructor.")
# print(tags)