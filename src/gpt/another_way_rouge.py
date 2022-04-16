from statistics import mean
from nltk import word_tokenize
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.gleu_score import corpus_gleu
from nltk.translate.meteor_score import meteor_score
from nltk.translate.bleu_score import SmoothingFunction
from rouge import Rouge 


from sumeval.metrics.rouge import RougeCalculator


rouge = RougeCalculator(stopwords=False, lang="zh")

rouge_1 = rouge.rouge_n(
            summary="我 今 天 晚 上 必 须 回 家 吃 饭",
            references="晚 上 我 要 回 家 吃 饭",
            n=1)

rouge_2 = rouge.rouge_n(
            summary="我 今 天 晚 上 必 须 回 家 吃 饭",
            references="晚 上 我 要 回 家 吃 饭",
            n=2)

rouge_l = rouge.rouge_l(
            summary="我 今 天 晚 上 必 须 回 家 吃 饭",
            references="晚 上 我 要 回 家 吃 饭")

# You need spaCy to calculate ROUGE-BE

rouge_be = rouge.rouge_be(
            summary="我今天没有吃早饭",
            references=["说你今天吃没吃早饭", "吃早饭了没"])

print("ROUGE-1: {}, ROUGE-2: {}, ROUGE-L: {}, ROUGE-BE: {}".format(
    rouge_1, rouge_2, rouge_l, rouge_be
).replace(", ", "\n"))