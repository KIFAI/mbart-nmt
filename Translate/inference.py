import nltk
import itertools
import ctranslate2
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

class Translator():
    '''
    Allow detokenizing sequences in batchs

    **Performance tips**
    Below are some general recommendations to further improve performance. Many of these recommendations were used in the WNGT 2020 efficiency task submission.

    Set the compute type to "auto" to automatically select the fastest execution path on the current system
    Reduce the beam size to the minimum value that meets your quality requirement
    When using a beam size of 1, keep return_scores disabled if you are not using prediction scores: the final softmax layer can be skipped
    Set max_batch_size and pass a larger batch to *_batch methods: the input sentences will be sorted by length and split by chunk of max_batch_size elements for improved efficiency
    Prefer the "tokens" batch_type to make the total number of elements in a batch more constant
    Consider using {ref}translation:dynamic vocabulary reduction for translation
    
    <On CPU>

    Use an Intel CPU supporting AVX512
    If you are processing a large volume of data, prefer increasing inter_threads over intra_threads and use stream methods (methods whose name ends with _file or _iterable)
    Avoid the total number of threads inter_threads * intra_threads to be larger than the number of physical cores
    For single core execution on Intel CPUs, consider enabling packed GEMM (set the environment variable CT2_USE_EXPERIMENTAL_PACKED_GEMM=1)
    
    <On GPU>

    Use a larger batch size
    Use a NVIDIA GPU with Tensor Cores (Compute Capability >= 7.0)
    Pass multiple GPU IDs to device_index to execute on multiple GPUs
    '''
    def __init__(self, model_path=None, model_type='Ctranslate2', device='cuda', device_index=[0], inter_threads=8, intra_threads=1,
            max_length=200, batch_size=8):
        assert model_type in ['Ctranslate2', 'Pytorch']

        if model_type == 'Ctranslate2':
            if device == 'cuda':
                if not isinstance(device_index, list):
                    raise TypeError("Device index must be list type")
                else:
                    self.model = ctranslate2.Translator(model_path, inter_threads=1, intra_threads=8, device=device, device_index=device_index)
            elif device == 'cpu':
                self.model = ctranslate2.Translator(model_path, inter_threads=inter_threads, intra_threads=intra_threads, device=device, device_index=0)

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
