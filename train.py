import loader
import numpy as np
import _pickle as cPickle
import sys
import time
import torch

from model import BiLSTM
from utils import adjust_learning_rate


train_path = ".\\dataset\\eng.train"
test_path = ".\\dataset\\eng.testa"
pre_embed_path = ".\\glove.6B.100d.txt"
mapping_file = ".\\dataset\\map_data.map"
model_path = ".\\BiLSTM_CRF.pth"

train_sentences = loader.load_data(train_path, zeros=False)
test_sentences = loader.load_data(test_path, zeros=False)

loader.update_tag_scheme(train_sentences, 'iob')
loader.update_tag_scheme(test_sentences, 'iob')

dico_words_train = loader.word_mapping(train_sentences, lower=True)[0]
dico_words, word_to_id, id_to_word = loader.augment_with_pretrain(
    dico_words_train.copy(), pre_embed_path, words=None
)

dico_chars, char_to_id, id_to_char = loader.char_mapping(train_sentences)
dico_tags, tag_to_id, id_to_tag = loader.tag_mapping(train_sentences)

train_data = loader.pepare_dataset(
    train_sentences, word_to_id, char_to_id, tag_to_id
)
test_data = loader.pepare_dataset(
    test_sentences, word_to_id, char_to_id, tag_to_id
)

print("%i / %i sentences in train / test." % (
    len(train_data), len(test_data)))

all_word_embeds = {}
for i, line in enumerate(open(pre_embed_path, 'r', encoding='UTF-8')):
    s = line.strip().split()
    if len(s) == 101:
        all_word_embeds[s[0]] = np.array([float(i) for i in s[1:]])
word_embeds = np.random.uniform(-np.sqrt(0.06), np.sqrt(0.06), (len(word_to_id), 100))
for w in word_to_id:
    if w in all_word_embeds:
        word_embeds[word_to_id[w]] = all_word_embeds[w]
    elif w.lower() in all_word_embeds:
        word_embeds[word_to_id[w]] = all_word_embeds[w.lower()]

print('Loaded {} pretrained embeddings.'.format(len(all_word_embeds)))

with open(mapping_file, 'wb') as f:
    mapping = {
        'word_to_id': word_to_id,
        'tag_to_id': tag_to_id,
        'char_to_id': char_to_id,
        'word_embeds': word_embeds
    }
    cPickle.dump(mapping, f)

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
learning_rate = 0.015
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
losses = []
model.train(True)
count = 0
for epoch in range(1000):
    print("epoch: {}".format(epoch))
    loss = 0.0
    for i, index in enumerate(np.random.permutation(len(train_data))):
        count += 1
        data = train_data[index]
        model.zero_grad()

        sentence_in = data['words']
        sentence_in = torch.tensor(sentence_in, dtype=torch.long)
        # sentence_in.requires_grad=True
        
        cap_in = data['caps']
        cap_in = torch.tensor(cap_in, dtype=torch.long)
        # cap_in.requires_grad=True

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

        neg_log_likelihood = model.neg_log_likelihood(sentence_in, tags, chars2_mask, cap_in, chars2_length, d)
        loss += neg_log_likelihood.item() / len(data['words'])
        neg_log_likelihood.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), 5.0)
        optimizer.step()

        if count % len(train_data) == 0:
            adjust_learning_rate(optimizer, lr=learning_rate/(1+0.05*count/len(train_data)))
    
    losses.append(loss / len(train_data))
    print("loss: {}".format(loss / len(train_data)))
    if len(losses) >= 2:
        if abs((losses[-2] - losses[-1]) / losses[-1] ) < 0.01:
            break

model.train(False)
torch.save(model.state_dict, model_path)
x = torch.load(model_path)
model.load_state_dict(x())
model.eval()

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
