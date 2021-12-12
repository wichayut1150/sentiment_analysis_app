import numpy as np


class Tokenizer:

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def tokenize(self, data, max_len=128):
        input_ids = []
        attention_masks = []
        for i in range(len(data)):
            encoded = self.tokenizer.encode_plus(
                data[i],
                add_special_tokens=True,
                max_length=max_len,
                padding='max_length',
                return_attention_mask=True
            )
            input_ids.append(encoded['input_ids'])
            attention_masks.append(encoded['attention_mask'])
        return np.array(input_ids), np.array(attention_masks)
