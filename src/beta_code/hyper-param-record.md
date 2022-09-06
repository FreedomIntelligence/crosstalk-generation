### Finetune-parameters

| Model | Hyper-para                                                   |
| ----- | ------------------------------------------------------------ |
| LSTM Seq2seq   | (batch-size=64,epochs=100,dropout=0.25,embed-size=300,vocab-size=7446,hidden-size=256,seed=42) |
| GPT   | (batch-size=64, cls-id=0,  epochs=100, eps=1e-09, gradient-accumulation-steps=4, ignore-index=-100, lr=0.00015, max-grad-norm=2.0, max-len=256,  pad-id=1, patience=0, pretrained-model='CDial-GPT-LCCC-base',  seed=1234, sep-id=2, warmup-rate=0.1) |
| UniLM | (batch-size=8,adam-epsilon=1e-08,gradient-accumulation-steps=1.learning-rate=1e-05,max-seq-length=256,model-name-or-path='torch-unilm-model',num-train-epochs=100,seed=42,warmup-proportion=0.1,weight-decay=0.01) |
| T5    | (batch-size=24, cls-id=101, epochs=100, eps=1e-09, gradient-accumulation-steps=4, ignore-index=-100, lr=0.00015, max-grad-norm=2.0, max-len=256,  pad-id=0, patience=0, pretrained-model='t5-pegasus-torch',  seed=1234, sep-id=102,  warmup-rate=0.1) |
| GPT3  | (model=Davinci,batch-size=1,learning-rate-multiplier=0.1,n-epochs=4,prompt-loss-weight=0.1) |



## Inference-parameters

### 

| Model         | Hyper-para                                                   |
| ------------- | ------------------------------------------------------------ |
| LSTM Seq2seq           | (repetition-penalty=1.0,  seed=42,  seq-max-len=256, temperature=1,  topk=8, topp=0, utterance-max-len=64) |
| GPT           | (repetition-penalty=1.0, seed=1234,  seq-max-len=256, temperature=1,  topk=8, topp=0, utterance-max-len=64) |
| UniLM         | (repetition-penalty=1.0,  seed=42,  seq-max-len=256, temperature=1,  topk=8, topp=0, utterance-max-len=64) |
| T5            | (repetition-penalty=1.0,  seed=1234,  seq-max-len=256, temperature=1,  topk=8, topp=0, utterance-max-len=64) |
| CPM-Large     | (repetition-penalty=2.0,  seed=1234,  seq-max-len=256, temperature=1.2,  topk=1, topp=0.9, utterance-max-len=64) |
| Panggu-α      | (repetition-penalty=2.0,  seed=1234,  seq-max-len=256, temperature=1.4,  topk=8, topp=0.91, utterance-max-len=64) |
| Zhouwenwang   | (repetition-penalty=2.0,  seed=1234,  seq-max-len=256, temperature=1.2,  topk=2, topp=0.3, utterance-max-len=64, diverse-penalty=2.0) |
| GPT3 (GPT3-Davinci)  | (temperature=0.5,max-tokens=384,top-p=1,frequency-penalty=0.5,presence-penalty=0) |
| GPT3-fine-tuned-Davinci | (temperature=0.5,max-tokens=384,top-p=1,frequency-penalty=0.5,presence-penalty=0) |





