import sentencepiece as spm

# train sentencepiece model from `botchan.txt` and makes `m.model` and `m.vocab`
# `m.vocab` is just a reference. not used in the segmentation.
input_text = YOUR_TRAIN_TEXT_FILE
model_prefix = YOUR_MODEL_NAME
vocab_size = YOUR_DESIRED_VOCAB_SIZE
spm_command = f'--input={input_text} --model_prefix={model_prefix} --vocab_size={vocab_size}'
spm.SentencePieceTrainer.train(spm_command)

# makes segmenter instance and loads the model file (farsi.model)
sp = spm.SentencePieceProcessor()
sp.load(f'{model_prefix}.model')

# encode: text => id
print(sp.encode_as_pieces('سلام این یک محصول خوب و با کیفیت است'))
print(sp.encode_as_ids('سلام این یک محصول خوب و با کیفیت است'))

