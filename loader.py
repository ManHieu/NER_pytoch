from utils import zero_digits, iob2, create_dico, create_mapping
import model
import os
import re


def load_data(path, zeros):
    """
    Load sentences from path (data set). Sentences are separated by empty lines.
    You can replace all digits to zeros if you want.
    """
    sentences = []
    sentence = []
    for line in open(path, 'r', encoding='UTF-8'):
        line = zero_digits(line.rstrip()) if zeros else line.rstrip()
        if line:
            word = line.split()
            assert len(word) >=2
            sentence.append(word)
        else:
            if 'DOCSTART' not in sentence[0][0]:
                sentences.append(sentence)
            sentence = []
    
    if len(sentence) > 0:
        if 'DOCSTART' not in sentence[0][0]:
                sentences.append(sentence)
    
    return sentences

def update_tag_scheme(sentences, tag_scheme):
    """
    Check and update sentences tagging scheme to IOB2.
    Only input IOB1 and IOB2 schemes are accepted.
    """
    for i, sentence in enumerate(sentences):
        tags = [word[-1] for word in sentence]
        if not iob2(tags):
            raise Exception("Sentence {}: {} should be given in IOB format!".format(i, sentence))

        if tag_scheme == 'iob':
            for word, new_tag in zip(sentence, tags):
                word[-1] = new_tag
        else:
            raise Exception('Unknown tagging scheme!')

def word_mapping(sentences, lower):
    """
    Create a dictionary and a mapping of words, sorted by frequency.
    """
    words = []
    for sentence in sentences:
        words.append(word[0].lower() if lower else word[0] 
                    for word in sentence)
    
    dico = create_dico(words)
    dico['<UNK>'] = 1000000
    dico['<PAD>'] = 1000001
    dico = {k:v for k, v in dico.items() if v > 2}

    id_to_word, word_to_id = create_mapping(dico)
    return dico, word_to_id, id_to_word

def tag_mapping(sentences):
    """
    Create a dictionary and a mapping of tags, sorted by frequency.
    """
    tags = [[word[-1] for word in s] for s in sentences]
    
    dico = create_dico(tags)
    dico[model.START_TAG] = -1
    dico[model.STOP_TAG] = -2

    id_to_tag, tag_to_id = create_mapping(dico)
    return dico, tag_to_id, id_to_tag

def char_mapping(sentences):
    """
    Create a dictionary and mapping of characters, sorted by frequency.
    """
    chars = [''.join([w[0] for w in sentence]) for sentence in sentences]
    
    dico = create_dico(chars)
    dico['<PAD>'] = 1000000

    id_to_char, char_to_id = create_mapping(dico)
    return dico, char_to_id, id_to_char

def cap_feature(s):
    """
    Capitalization feature:
    0 = low caps
    1 = all caps
    2 = first letter caps
    3 = one capital (not first letter)
    """
    if s.lower() == s:
        return 0
    elif s.upper() == s:
        return 1
    elif s[0].upper() == s[0]:
        return 2
    else:
        return 3

def padding(seq, max_len, PAD_TOKEN=0):
    seq += [PAD_TOKEN for i in range(max_len - len(seq))]
    return seq

def augment_with_pretrain(dictionary, ext_emb_path, words):
    """
    Augment the dictionary with words that have a pretrained embedding.
    If `words` is None, we add every word that has a pretrained embedding
    to the dictionary, otherwise, we only add the words that are given by
    `words` (typically the words in the development and test sets.)
    """
    assert os.path.isfile(ext_emb_path)
    pretrain = set([
        line.rstrip().split()[0].strip()
        for line in open(ext_emb_path, 'r', encoding="UTF-8")
    ])

    if words is None:
        for word in pretrain:
            if word not in dictionary:
                dictionary[word] = 0

    else:
        for word in words:
            if any(x in pretrain for x in [word, word.lower(), re.sub(r'\d', '0', word.lower())]) and word not in dictionary:
                dictionary[word] = 0    

    id_to_word, word_to_id = create_mapping(dictionary)
    return dictionary, word_to_id, id_to_word

def pepare_dataset(sentences, word_to_id, char_to_id, tag_to_id, lower=True):
    def f(x):
        return x.lower() if lower else x

    data = []
    for sentence in sentences:
        str_words = [word[0] for word in sentence]
        tags = [tag_to_id[word[-1]] for word in sentence]
        words = []
        chars = []
        caps = []
        for word in str_words:
            word_id = word_to_id[f(word) if f(word) in word_to_id else '<UNK>']
            words.append(word_id)

            char_id = [char_to_id[c] for c in word if c in char_to_id]
            chars.append(char_id)

            caps.append(cap_feature(word))
        
        data.append({
            'str_words': str_words,
            'words': words,
            'chars': chars,
            'caps': caps,
            'tags':tags
        })
    return data
        
def prepare_sentence(str_words, word_to_id, char_to_id, lower=False):
    def f(x): return x.lower() if lower else x

    words = []
    chars = []
    caps = []
    for word in str_words:
        word_id = word_to_id[f(word) if f(word) in word_to_id else '<UNK>']
        words.append(word_id)

        char_id = [char_to_id[c] for c in word if c in char_to_id]
        chars.append(char_id)

        caps.append(cap_feature(word))
    
    return {
        'str_words': str_words,
        'words': words,
        'chars': chars,
        'caps': caps
    }

