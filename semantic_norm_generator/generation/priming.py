def generate_single_prime_sentence(train_df, question):
    priming_text = ''
    for row in train_df:
        priming_text += row[0]
        priming_text += ' '
        priming_text += row[1]

    text = priming_text + '\n%s' % question
    return text