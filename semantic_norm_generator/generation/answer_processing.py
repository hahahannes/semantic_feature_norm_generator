def escape_answer(self, text):
    # newlines are useless, double quotes are used in the output CSV, so need be to escaped or removed
    return text.replace('\n', '').replace('"', '""')
