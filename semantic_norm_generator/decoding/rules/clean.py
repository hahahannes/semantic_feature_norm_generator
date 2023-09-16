from semantic_norm_generator.decoding import helper as helper


def rule_remove_properties_quesion(feature, concept, nlp):
    if 'What are the properties' in feature:
        return False
    return True

def rule_remove_underscores(feature, concept):
    if '_' in feature:
        return False
    return True

def rule_remove_one_word_features(feature, concept, nlp):
    list_of_words = helper.tokenize(feature)
    tagged_feature = nlp(feature)

    if len(list_of_words) == 1 and not tagged_feature[0].tag_.startswith('V') and tagged_feature[0].lemma_ != 'be':
        return False
    elif len(list_of_words) == 0:
        return False
    return True

def rule_remove_question_marks(feature, concept, nlp):
    if '?' in feature:
        return False
    return True

def rule_remove_not_pronom(feature, concept, nlp):
    tagged_feature = nlp(feature)
    if tagged_feature[0].tag_ != 'PRP':
        return False
    return True

def rule_remove_non_ascii(feature, concept, nlp):
    if not feature.isascii():
        return False
    return True

def remove_same_noun(feature, concept, nlp):
    tagged_feature = nlp(feature)
    if len(tagged_feature) >= 3:
        if tagged_feature[0].lemma_ == 'be' and tagged_feature[2].text == concept:
            return False 
    elif len(tagged_feature) >= 2:
        if tagged_feature[0].lemma_ == 'be' and tagged_feature[1].text == concept:
            return False 
    return True
