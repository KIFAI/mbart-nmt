import time
from transformers import (MBartForConditionalGeneration, MBartTokenizer)

def generate(model, result_path, sentence_bucket, batch_size = 16, cuda_id = 0, half_precision=True):
    if half_precision:
        model.half()
    tokenizer = MBartTokenizer.from_pretrained(result_path)
    def batch(iterable, n=1):
        l = len(iterable)
        for ndx in range(0, l, n):
            yield iterable[ndx:min(ndx + n, l)]
    sentence_batch = batch(sentence_bucket, batch_size)
    hyps = []
    start_time = time.time()
    for sents in sentence_batch:
        inputs = tokenizer(sents, return_tensors="pt", padding=True).to(f'cuda:{cuda_id}')
        translated_tokens = model.generate(
            **inputs, decoder_start_token_id=tokenizer.lang_code_to_id["ko_KR"], num_beams=2, early_stopping=True, max_length=150)
        pred = tokenizer.batch_decode(
            translated_tokens, skip_special_tokens=True)
        hyps.extend(pred)  
    end_time = time.time()
    print(f'Elapsed time : {end_time-start_time}')
    total_tokens = sum([len(s.split()) for s in sentence_bucket])
    avg_tokens = total_tokens/len(sentence_bucket)
    print(
        f'Total sentence : {len(sentence_bucket)}\nTotal tokens : {total_tokens}\nAvg tokens : {avg_tokens}')
    return hyps

sources = [':46A: DOCUMENTS REQUIRED',
           '+',
           'SIGNED COMMERCIAL INVOICE IN 3 ORIGINAL(S) AND 3 COPY(IES)',
           'INDICATING CREDIT NUMBER.',
           '+FULL SET OF CLEAN ON BOARD BILL OF LADING MADE OUT TO THE ORDER OF CHANG HWA COMMERCIAL BANK, LTD.',
           "MARKED ''FREIGHT PREPAID'' AND CREDIT NUMBER AND NOTIFY APPLICANT WITH FULL ADDRESS.",
           '+FULL SET OF INSURANCE POLICY OR CERTIFICATE, FORAT LEAST 110% OF INVOICE VALUE, BLANK ENDORSED AND WITH CLAIMS PAYABLE IN TAIWAN, COVERING:',
           'INSTITUTE CARGO CLAUSES',
           '(A).',
           '+PACKING LIST IN 3',
           'ORIGINAL(S) AND 3 COPY(IES).',
           ":47A: ADDITIONAL CONDITIONS",
           "+AN EXTRA COPY OF ALL DOCUMENTS IS REQUIRED FOR ISSUING BANK'S FILE.",
           'OTHERWISE, A FEE OF USD10(JPY1100.00 OR EUR10.00) OR EQUIVALENT AMOUNT IN L\\/C CURRENCY WILL BE DEDUCTED FROM THE PROCEEDS.',
           '+DOCUMENTS MUST BE PRESENTED TO ISSUINGBANK THROUGH A BANK ONLY.',
           '+ALL DOCUMENTS MUST BE MADE IN ENGLISH UNLESS OTHERWISE STIPULATED IN CREDIT.',
           'HOWEVER, DOCUMENTS IN LANGUAGES OTHER THAN ENGLISH ARE ACCEPTABLE PROVIDED THEY ALSO BEAR THE TEXT IN ENGLISH LANGUAGE.',
           '+SHIPMENT MUST BE CONTAINERIZED.',
           '+10% MORE OR LESS IN AMOUNT IS ACCEPTABLE.',
           '+ALL DOCUMENTS INCLUDING FREE SAMPLE ARE ACCEPTABLE.',
           "+IF WE GIVE OUR NOTICE OF REFUSAL DUE TO DOCUMENTS WITH DISCREPANCIES WE SHALL HOLD DOCUMENTS AT THE PRESENTER'S RISK AND DISPOSAL.",
           "HOWEVER IF WE HAVE NOT RECEIVED THE PRESENTER'S DISPOSAL INSTRUCTIONS FOR THE DISCREPANT DOCUMENTS PRIOR TO RECEIPT OF THE APPLICANT'S WAIVER OF DISCREPANCIES TO OUR SATISFACTION, WE SHALL RELEASE THE DOCUMENTSTO THE APPLICANT AND ARRANGE PAYMENT WITHOUT FURTHER NOTICE TO THE PRESENTER.",
           'UNDER NO CIRCUMSTANCES SHALL WE ACCEPT ANY LIABILITY FOR SUCH RELEASE.',
           '+BANKING HOU.S. WE, CHANG HWA COMMERCIAL BANK, LTD.',
           "PURSUANT TO THE ARTICLE 33 ''HOURS OF PRESENTATION'' OF UCP600, WILL PRINCIPALLY ACCEPT A PRESENTATION PERIOD FOR THE FOLLOWING:",
           'DOCUMENTS RECEIVED AFTER',
           '2:00 P.M.',
           'LOCAL TIME AT OUR COUNTER WILL BE CONSIDERED ASPRESENTED ON THE NEXT BANKING DAY.',
           '+ALL PARTIES TO THIS TRANSACTION ARE ADVISED THAT BANKS MAY BE UNABLE TO PROCESS ANY DOCUMENTS, SHIPMENTS, GOODS, PAYMENT AND\\/OR TRANSACTIONS THAT MAY RELATE, DIRECTLY OR INDIRECTLY, TO ANY COUNTRIES, REGIONS, ENTITIES, VESSELS OR INDIVIDUALS SANCTIONED BY THE UNITED NATIONS, THE UNITED STATES, THE EUROPEAN UNION, THE UNITED KINGDOM OR ANY OTHER RELEVANT GOVERNMENT AND\\/OR REGULATORY AUTHORITIES MAY REQUIRE DISCLOSURE OF INFORMATION WHICH THE BANK MAY HAVE TO COMPLY WITH\\/OR WITHOUT INFORMING YOU.',
           'CHANG HWA COMMERCIAL BANK, LTD.',
           'IS NOT LIABLE FOR ANY LOSS, DAMAGE OR COST INCURRED IF IT, OR ANY OTHER PERSON, FAILS OR DELAYS TO PERFORM THE TRANSACTION OR DISCLOSES INFORMATION AS A RESULT OF ACTUAL OR POTENTIAL BREACH OF SUCH SANCTION.',
           ':71D: CHARGES',
           "+ALL BANKING CHARGES EXCEPT THIS L\\/C ISSUING CHARGES ARE FOR BENEFICIARY'S ACCOUNT.",
           ':48 : PERIOD FOR PRESENTATION IN DAYS',
           '021',
           ':78 : INSTRUCTIONS TO THE PAYING\\/ACCEPTING\\/NEGOTIATING',
           '+A DISCREPANCY FEE OF USD70.00(JPY7700.00 OR EUR70.00) OR EQUIVALENT AMOUNT IN L\\/C CURRENCY WILL BEDEDUCTED FROM THE REIMBURSEMENT CLAIM \\/ PROCEEDS UPON EACH PRESENTATION OF DISCREPANT DOCUMENTS.',
           '+DOCUMENTS ARE TO BE FORWARDED TO US:',
           '(2F, NO.57, SEC.2, CHUNG SHAN NORTH ROAD, TAIPEI 10412 TAIWAN TEL:886-2-25362951) IN ONE SET BY COURIER SERVICE.',
           '+IF RECEIVING THE SHIPPING DOCUMENTS IN COMPLIANCE WITH THE TERMS AND CONDITIONS, WE SHALL EFFECT THE PAYMENT (LESS WIRE TRANSFER\\/HANDLING CHARGES) AS PER YOUR INSTRUCTIONS.']

cuda_id = 0
reduction_fp16_path = '/opt/project/translation/mbart-nmt/src/ftm/mbart-fp16-finetuned-en_XX-to-ko_KR/final_checkpoint'
fp16_model = MBartForConditionalGeneration.from_pretrained(reduction_fp16_path).to(f"cuda:{cuda_id}")
# reduction_fp32_path = '/opt/project/translation/mbart-nmt/src/ftm/mbart-fp32-finetuned-en_XX-to-ko_KR/final_checkpoint'
# fp32_model = MBartForConditionalGeneration.from_pretrained(reduction_fp32_path).to(f"cuda:{cuda_id}")

hyps = generate(fp16_model, reduction_fp16_path, sources, batch_size=16, cuda_id=cuda_id, half_precision=True)
# hyps = generate(fp32_model, reduction_fp32_path, sources, cuda_id, half_precision=False)
for src, hyp in zip(sources, hyps):
    print(f"src:{src}\nhyp:{hyp}")


