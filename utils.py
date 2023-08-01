import sys
import lmppl
sys.path.insert(0, "/content/lm-watermarking")
from watermark_processor import WatermarkDetector, WatermarkLogitsProcessor

scorer = lmppl.EncoderDecoderLM('google/flan-t5-small')

def get_perplexity(input_sentences, output_sentences):
  ppl = scorer.get_perplexity(input_texts=input_sentences, output_texts=output_sentences)
  return ppl

class WatermarkDetectorClass():
  def __init__(self, tokenizer, gamma, device): 
    self.watermark_detector = WatermarkDetector(vocab=list(tokenizer.get_vocab().values()),
                                            gamma=gamma, # should match original setting
                                            seeding_scheme="simple_1", # should match original setting
                                            device=device, # must match the original rng device type
                                            tokenizer=tokenizer,
                                            z_threshold=4.0,
                                            normalizers=[],
                                            ignore_repeated_bigrams=False)
  def detect(self, text):
    score_dict = self.watermark_detector.detect(text) # or any other text of interest to analyze
    return score_dict