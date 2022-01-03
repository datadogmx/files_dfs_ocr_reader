import pickle

class TextDFSEvaluator:
    def __init__(self, word_vectorizer_path="models/word_vectorizer.pckl", word_integrity_path='models/word_integrity.pckl'):
        self.word_vectorizer = pickle.load(open(word_vectorizer_path, 'rb'))
        self.random_cls = pickle.load(open(word_integrity_path, 'rb'))

    def evaluate(self, data):
        try:
            x_test = self.word_vectorizer.transform(data).toarray()
            y_pred = self.random_cls.predict(x_test)
            y_prob = self.random_cls.predict_proba(x_test)[:,1]
            return y_pred, y_prob
        except:
            return None, None
