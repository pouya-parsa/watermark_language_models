import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import (LogitsProcessorList)
import sys
sys.path.insert(0, "/content/lm-watermarking")
from watermark_processor import WatermarkDetector, WatermarkLogitsProcessor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = T5ForConditionalGeneration.from_pretrained('t5-small').to(device)
tokenizer = T5Tokenizer.from_pretrained('t5-small', model_max_length=3072)

def summarize(text, watermark=False, gamma=0.25, delta=2.0):
  if not watermark:
    inputs = tokenizer(["summarize: " + text], return_tensors="pt").to(device)
    summary = model.generate(**inputs, return_dict_in_generate=True, max_new_tokens=1000, output_scores=True)
    summary = tokenizer.batch_decode(summary.sequences, skip_special_tokens=True)[0]
  else:
    watermark_processor = WatermarkLogitsProcessor(vocab=list(tokenizer.get_vocab().values()),
                                               gamma=gamma,
                                               delta=delta,
                                               seeding_scheme="simple_1")
    tokenized_input = tokenizer(["summarize: " + text], return_tensors="pt").to(device)
    output_tokens = model.generate(**tokenized_input,
                                  logits_processor=LogitsProcessorList([watermark_processor]), max_new_tokens=1000, return_dict_in_generate=True, output_scores=True)
    summary = tokenizer.batch_decode(output_tokens.sequences, skip_special_tokens=True)[0]
   
  return summary