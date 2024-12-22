from sentence_transformers import SentenceTransformer as ST

class SentenceTransformer:
    def __init__(self, model="all-MiniLM-L6-v2"):
        self.model = ST(model)

    def encode(self, text):
        encoded = self.model.encode(text)
        return encoded
    
    def decode(self, text):
        decoded = self.model.decode(text)
        return decoded
    
if __name__ == '__main__':
    import numpy as np
    sentences = [
        "The weather is lovely today.",
        "It's so sunny outside!",
        "He drove to the stadium.",
    ]
    
    model = SentenceTransformer("all-MiniLM-L6-v2")
    encoded1 = model.encode(sentences[0])
    encoded2 = model.encode(sentences[1])
    encoded3 = model.encode(sentences[2])
    
    similarity12 = np.dot(encoded1, encoded2) / (np.linalg.norm(encoded1) * np.linalg.norm(encoded2))
    similarity13 = np.dot(encoded1, encoded3) / (np.linalg.norm(encoded1) * np.linalg.norm(encoded3))
    print(similarity12, similarity13)