
import sentencepiece as spm
''' Merge tokenizer '''
import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"]="python"
from sentencepiece import sentencepiece_model_pb2 as model
from transformers import  AutoTokenizer
save_vocab_path = PATH_FOR_MERGED_TOKENIZER
orig_model_path = PATH_FOR_SOURCE_MODEL_TOKENIZER
farsi_model_path = PATH_FOR_FARSI_MODEL_TOKENIZER
model_name = YOUR_SOURCE_MODEL
orig_m = model.ModelProto()
farsi_m = model.ModelProto()
orig_m.ParseFromString(open(orig_model_path, "rb").read())
farsi_m.ParseFromString(open(farsi_model_path, "rb").read())
print(len(orig_m.pieces), len(farsi_m.pieces))
orig_pieces = []
for piece in orig_m.pieces:
    orig_pieces.append(piece.piece)
for piece in farsi_m.pieces:
    if piece.piece not in orig_pieces:
        orig_m.pieces.append(piece)
        orig_pieces.append(piece.piece)

print(len(orig_m.pieces))
with open(save_vocab_path, 'wb') as f:
    f.write(orig_m.SerializeToString())
## Test Tokenizer
mistral_tokenizer = AutoTokenizer.from_pretrained(model_name)
path = DIRECTORY_FOR_MERGED_TOKENIZER
farsi_mistral_tokenizer = AutoTokenizer.from_pretrained(DIRECTORY_FOR_MERGED_TOKENIZER)
print(farsi_mistral_tokenizer.all_special_tokens)
print(farsi_mistral_tokenizer.all_special_ids)
print(farsi_mistral_tokenizer.special_tokens_map)
print(len(farsi_mistral_tokenizer))
text='''
چگونه می‌توانیم یک ایده خوب را به محصولی پولساز و واقعی تبدیل کنیم.
'''
print("Test text:\n",text)
print(f"Tokenized by Mistral tokenizer:{mistral_tokenizer.tokenize(text)}")
print("Tokenized by Farsi-Mistral tokenizer",farsi_mistral_tokenizer.tokenize(text))