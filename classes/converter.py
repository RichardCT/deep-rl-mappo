import utils.req as req
from classes.sentence_transformer import SentenceTransformer

class EnvironmentConverter():
    def __init__(self):
        self._sentence_transformer = SentenceTransformer()
    
    def convert(self, render):
        env = self._llava(render)
        env = self._llama(env)
        env = self._transform(env)
        return env
    
    def _llava(self, image):
        return req.get_llava(image)
    
    def _llama(self, text):
        return req.get_llama(text)
    
    def _transform(self, text):
        return self._sentence_transformer.convert(text)
        