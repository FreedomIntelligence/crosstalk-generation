from transformers.tokenization_utils import PreTrainedTokenizer

import torch
import sentencepiece
import jieba


class GPTPanguTokenizer(PreTrainedTokenizer):
    # Ref: https://git.openi.org.cn/PCL-Platform.Intelligence/PanGu-Alpha/src/branch/master/tokenization_jieba.py
    vocab_files_names = {
        "model_file": "vocab.model"
    }

    def __init__(
            self,
            model_file,
            **kwargs
    ):
        super().__init__()

        self.sp = sentencepiece.SentencePieceProcessor()
        self.sp.Load(model_file=model_file)
        self.translator = str.maketrans(" \n", "\u2582\u2583")

        # special token ids
        self.eos_token_id = self.sp.piece_to_id("<eot>")

    def tokenize(self, text, **kwargs):
        """ Tokenize a string. """
        seg_list = [x.translate(self.translator) for x in jieba.cut(text, cut_all=False)]
        new_seg = " ".join(seg_list)
        return self.sp.encode(new_seg)

    def convert_tokens_to_ids(self, tokens):
        return tokens

    def convert_ids_to_tokens(self, ids):
        return self.decode(ids)

    def decode(self, tokens, **kwargs):
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()

        text = self.sp.decode(tokens)
        text = text.replace(' ', '').replace('\u2582', ' ').replace('\u2583', '\n')
        return text
