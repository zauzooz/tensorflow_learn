import numpy as np

vocab_list = {
    "orange" : [0, 4, 0],
    "apple": [3, 3, 0],
    "phone": [4, 0, 0],
    "an": [0, 0, 2],
    "and": [0, 0 , 1]
}

def simple_embedding(sentence: str, max_len: int):
    sentence = sentence.lower()
    result =  [vocab_list[word] for word in sentence.split(" ")]
    if len(result) < max_len:
        result += [[1, 1, 1]]*(max_len - len(result))
    return np.array(result)