import os
import pickle

def str2index(str, charset):
    # this_path = os.path.dirname(os.path.realpath(__file__))
    # charpkl = pickle.load(open(os.path.join(this_path, 'char.pkl'),
    #                            'rb'))
    if charset == 'ENcap':
        # charset_ = charpkl['ENc']
        charset_ = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
                    'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R',
                    'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
        charsetstr = ''.join(charset_)
    elif charset == 'ENfull':
        # charset_ = charpkl['EN'] + charpkl['PUNC']
        charset_ = ['A', 'a', 'B', 'b', 'C', 'c', 'D', 'd', 'E',
                    'e', 'F', 'f', 'G', 'g', 'H', 'h', 'I', 'i',
                    'J', 'j', 'K', 'k', 'L', 'l', 'M', 'm', 'N',
                    'n', 'O', 'o', 'P', 'p', 'Q', 'q', 'R', 'r',
                    'S', 's', 'T', 't', 'U', 'u', 'V', 'v', 'W',
                    'w', 'X', 'x', 'Y', 'y', 'Z', 'z']
        charset_ = charset_ + ['.', ',', ';', ':', '!', '?', "'", '"']
        charsetstr = ''.join(charset_)
    l = []
    for char in str:
        # print(char)
        l = l + [charsetstr.find(char)]
        # print(l)
    return l

def char2index(char, charset):
    # this_path = os.path.dirname(os.path.realpath(__file__))
    # charpkl = pickle.load(open(os.path.join(this_path, 'char.pkl'),
    #                            'rb'))
    if charset == 'ENcap':
        # charset_ = charpkl['ENc']
        charset_ = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
                    'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R',
                    'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
        charsetstr = ''.join(charset_)
    elif charset == 'ENfull':
        # charset_ = charpkl['EN'] + charpkl['PUNC']
        charset_ = ['A', 'a', 'B', 'b', 'C', 'c', 'D', 'd', 'E',
                    'e', 'F', 'f', 'G', 'g', 'H', 'h', 'I', 'i',
                    'J', 'j', 'K', 'k', 'L', 'l', 'M', 'm', 'N',
                    'n', 'O', 'o', 'P', 'p', 'Q', 'q', 'R', 'r',
                    'S', 's', 'T', 't', 'U', 'u', 'V', 'v', 'W',
                    'w', 'X', 'x', 'Y', 'y', 'Z', 'z']
        charset_ = charset_ + ['.', ',', ';', ':', '!', '?', "'", '"']
        charsetstr = ''.join(charset_)
    return charsetstr.find(char)
