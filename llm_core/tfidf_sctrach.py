import math


def calculateTFIDF(docs: list):
    docs_count = len(docs)
    tokenized_docs = [doc.strip().split() for doc in docs]
    words = set(word for doc in tokenized_docs for word in doc)

    word_doc_freq = {}
    
    for word in words:
        contains_word = sum(1 for doc in tokenized_docs if word in doc)
        word_doc_freq[word] = contains_word

    tfidf_vectors = []

    for doc in tokenized_docs:
        doc_tfidf = {}
        total_word_in_doc = len(doc)
        for word in set(doc):
            tf = doc.count(word) / total_word_in_doc

            idf = math.log(docs_count/(word_doc_freq[word]))

            doc_tfidf[word] = tf*idf

        tfidf_vectors.append(doc_tfidf)

    return tfidf_vectors


docs = ["the cat sat", "the dog barked", "the cat chased the dog"]
print(calculateTFIDF(docs))