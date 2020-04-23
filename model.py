import torch
import torch.autograd as autograd
import torch.nn as nn


START_TAG = '<START>'
STOP_TAG = '<STOP>'

def to_scalar(var):
    return var.view(-1).data.tolist()[0]

class BiLSTM(nn.Module):
    def __init__(self, voca_size, word_emb_dim, pre_word_emb,
                char_emb_dim, char_lstm_dim, char_to_ix, 
                n_cap, cap_emb_dim, hidden_dim,
                tag_to_ix):
        super(BiLSTM, self).__init__()
        self.word_emb_dim = word_emb_dim
        self.char_emb_dim = char_emb_dim
        self.char_lstm_dim = char_lstm_dim
        self.cap_emb_dim = cap_emb_dim
        self.target_size = len(tag_to_ix)
        self.hidden_dim = hidden_dim

        self.char_embs = nn.Embedding(len(char_to_ix), self.char_emb_dim)
        self.char_lstm = nn.LSTM(self.char_emb_dim, self.char_lstm_dim, bidirectional=True)

        self.cap_embs = nn.Embedding(n_cap, self.cap_emb_dim)

        self.word_embs = nn.Embedding(voca_size, self.word_emb_dim)
        self.word_embs.weight = nn.Parameter(torch.tensor(pre_word_emb, dtype=torch.float))

        self.lstm = nn.LSTM(word_emb_dim+char_lstm_dim*2+cap_emb_dim, hidden_dim, bidirectional=True)
        self.hidden_to_tag = nn.Linear(hidden_dim*2, self.target_size)
        self.dropout = nn.Dropout(0.5)

    def _get_lstm_features(self, sentence, chars2, caps, chars2_length, matching_char):
        chars_embeds = self.char_embs(chars2).transpose(0, 1)
        packed = torch.nn.utils.rnn.pack_padded_sequence(chars_embeds, chars2_length)
        lstm_out_char, _ = self.char_lstm(packed)
        outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(lstm_out_char)
        outputs = outputs.transpose(0, 1)
        chars_embeds_temp = torch.tensor(torch.zeros((outputs.size(0), outputs.size(2))), dtype=torch.long)

        for i, index in enumerate(output_lengths):
            chars_embeds_temp[i] = torch.cat((outputs[i, index-1, :self.char_lstm_dim], outputs[i, 0, self.char_lstm_dim:]))
        chars_embeds = chars_embeds_temp.clone()
        for i in range(chars_embeds.size(0)):
            chars_embeds[matching_char[i]] = chars_embeds_temp[i]

        word_embeds = self.word_embs(sentence)

        cap_embeds = self.cap_embs(caps)

        embeds = torch.cat((word_embeds, chars_embeds.float(), cap_embeds), 1)
        embeds = embeds.unsqueeze(1)
        embeds = self.dropout(embeds)
        lstm_out, _ = self.lstm(embeds)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim*2)
        lstm_out = self.dropout(lstm_out)
        lstm_feats = self.hidden_to_tag(lstm_out)
        return lstm_feats

    def neg_log_likelihood(self, sentence, tags, chars2, caps, chars2_length, matching_char):
        feats = self._get_lstm_features(sentence, chars2, caps, chars2_length, matching_char)
        scores = nn.functional.cross_entropy(feats, tags)
        return scores
    
    def forward(self, sentence, chars, caps, chars2_length, matching_char):
        feats = self._get_lstm_features(sentence, chars, caps, chars2_length, matching_char)
        score, tag_seq = torch.max(feats, 1)
        tag_seq = list(tag_seq.cpu().data)

        return score, tag_seq
        