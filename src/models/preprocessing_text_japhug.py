MISC_SYMBOLS = [' ̩', '~', '=', ':', '¨', '↑', '“', '”', '…', '«', '»', 'ː', '#', '$']
PUNC_SYMBOLS = [',', '!', '.', ';', '?', "'", '"', '*', ':', '«', '»', '“', '”', "ʔ", "+", "-", "_", '\ufeff']


def preprocessing_text(sentence):
    sentence = sentence.replace('）', ')')
    if sentence[0] in MISC_SYMBOLS:
        # We assume these symbols cannot be captured.
        return None, sentence[1:]
    if sentence[0] in PUNC_SYMBOLS:
        return None, sentence[1:]
    if sentence[0] == "/":
        # We keep everything literal, thus including what is in /.../ ; so we just remove these tokens"
        return None, sentence[1:]
    if sentence[0] == "(":
        if sentence.find(")") == len(sentence) - 1:
            return None, ""
        else:
            return None, sentence[sentence.find(")") + 1:]
    if sentence[0] in {" ", "\t", "\n"}:
        # Return a space char so that it can be identified in word segmentation
        # processing.
        return " ", sentence[1:]
    return sentence[0], sentence[1:]


def filter_for_phonemes(sentence):
    """ Returns a sequence of phonemes and pipes (word delimiters). Tones,
        syllable boundaries, whitespace are all removed."""
    filtered_sentence = []
    phonemes = []
    while sentence != "":
        phoneme, sentence = preprocessing_text(sentence)
        filtered_sentence.append(phoneme)
        phonemes.append(phoneme)
    filtered_sentence = [item for item in filtered_sentence if item != None]
    return "".join(filtered_sentence), phonemes


def final_text_phonemes(batch):
    if "BEGAIEMENT" in batch['sentence']:
        batch['sentence'] = batch['sentence'].replace('BEGAIEMENT', "")
    s, p = filter_for_phonemes(batch['sentence'])
    s = s.replace(' ', '|')
    s = s.replace('||', '|')
    s = s.replace('|||', '|')
    batch['sentence'] = s
    p = list(filter(None, p))
    batch['phonemes'] = p
    return batch


def final_text_words(batch):
    s, p = filter_for_phonemes(batch['sentence'])
    batch['sentence'] = s
    # print(batch['sentence'])
    # print(batch['sentence'])
    # p = list(filter(None, p))
    # batch['phonemes'] = p
    return batch
