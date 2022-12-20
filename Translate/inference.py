import nltk
import itertools
import ctranslate2
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

class Translator():
    '''
    Allow detokenizing sequences in batchs
    '''
    def __init__(self, model_path=None, model_type='Ctranslate2', device='cuda', device_index=[0],
            max_length=200, batch_size=8):
        assert model_type in ['Ctranslate2', 'Pytorch']

        if model_type == 'Ctranslate2':
            if device == 'cuda':
                if not isinstance(device_index, list):
                    raise TypeError("Device index must be list type")
                else:
                    self.model = ctranslate2.Translator(model_path, inter_threads=1, intra_threads=8, device=device, device_index=device_index)
            elif device == 'cpu':
                device_index=None
                self.model = ctranslate2.Translator(mode_path, inter_threads=8, intra_threads=1, device=device, device_index=device_index)

        self.tokenizer = MBart50TokenizerFast.from_pretrained(model_path)
        self.max_length = max_length
        self.batch_size = batch_size

    def do_reassemble(self, tokenized_sents):
        '''
        Recombine sentences according to max length
        '''
        input_len, start_ix = 0, 0
        segments = []

        for i, t_s in enumerate(tokenized_sents):
            input_len += len(t_s)
            if i+1 == len(tokenized_sents):
                segments.append([self.tokenizer.src_lang] + list(itertools.chain(*tokenized_sents[start_ix:])) + [self.tokenizer.eos_token])
            elif input_len + len(tokenized_sents[i+1]) > self.max_length:
                segments.append([self.tokenizer.src_lang] + list(itertools.chain(*tokenized_sents[start_ix:i+1])) + [self.tokenizer.eos_token])
                input_len = 0
                start_ix = i+1
            else:
                pass
        return segments

    def convert_to_inputs(self, src_sent):
        '''
        Create Ctranslate input format according to the number of sent element
        '''
        splitted_sents = nltk.sent_tokenize(src_sent)
        print(f"\nSplitted Length of input : {len(splitted_sents)}")
        if len(splitted_sents) == 1:
            print(f"Do simply segmentation by max decoding length")

            def devide_inputs(l, n):
                for i in range(0, len(l), n):
                    yield [self.tokenizer.src_lang] + l[i:i+n] + [self.tokenizer.eos_token]

            tokenized_sent = self.tokenizer.tokenize(src_sent)
            segments = list(devide_inputs(tokenized_sent, self.max_length))

            print(f"Divided input's length : {len(segments)}")
            print(f"segments : {segments}")
            return segments
        else:
            return self.do_reassemble(list(map(self.tokenizer.tokenize, splitted_sents)))

    def detokenize(self, translated_tokens):
        '''
        Detokenize translated tokens
        '''
        return ' '.join(list(map(lambda x : self.tokenizer.convert_tokens_to_string(x.hypotheses[0][1:]).replace('<unk>', ''), translated_tokens)))

    def generate(self, src_sents, src_lang, tgt_lang):
        '''
        Main function of batch generation
        '''
        self.tokenizer.src_lang = src_lang

        def batch(iterable, n=1):
            l = len(iterable)
            for ndx in range(0, l, n):
                yield iterable[ndx:min(ndx + n, l)]
        
        sentence_batch = batch(src_sents, self.batch_size) if isinstance(src_sents, list) else batch([src_sents], self.batch_size)
        results = []

        for i, src_sent in enumerate(sentence_batch):
            inputs = list(map(self.convert_to_inputs, src_sent))
            translated_tokens = map(lambda source : self.model.translate_batch(source=source, target_prefix=[[tgt_lang]]*len(source) ,beam_size=2, max_decoding_length=self.max_length, asynchronous=False), inputs)
            pred = list(map(self.detokenize, translated_tokens))
            results.append(pred)

        return results
