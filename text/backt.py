from transformers import MarianMTModel, MarianTokenizer

fr_model_name = 'Helsinki-NLP/opus-mt-en-ROMANCE'
fr_tokenizer = MarianTokenizer.from_pretrained(fr_model_name)
fr_model = MarianMTModel.from_pretrained(fr_model_name)

en_model_name = 'Helsinki-NLP/opus-mt-ROMANCE-en'
en_tokenizer = MarianTokenizer.from_pretrained(en_model_name)
en_model = MarianMTModel.from_pretrained(en_model_name)


def translate(texts, model, tokenizer, language):
    # format text for nmt
    template = lambda text: f"{text}" if language == "en" else f">>{language}<< {text}"
    src_texts = [template(text) for text in texts]
    # tokenize
    encoded = tokenizer.prepare_seq2seq_batch(src_texts)
    # generate translation
    translated = model.generate(**encoded)
    # itos
    translated_texts = tokenizer.batch_decode(translated, skip_special_tokens=True)
    return translated_texts


def back_translate(texts, source_lang="fr", target_lang="en"):
    # source >> target
    translated_texts = translate(texts, en_model, en_tokenizer, language=target_lang)
    # target >> source
    back_translated_texts = translate(translated_texts, fr_model, fr_tokenizer, language=source_lang)
    return back_translated_texts


en_texts = ["ce n'est pas vraiement de l'anglais mais bon tant pis", "ah sinon oui j'ai faim"]
aug_texts = back_translate(en_texts, source_lang="fr", target_lang="en")
