import lmppl

scorer = lmppl.EncoderDecoderLM('google/flan-t5-large')

def get_perplexity(input_sentences, output_sentences):
  ppl = scorer.get_perplexity(input_texts=input_sentences, output_texts=output_sentences)
  return ppl