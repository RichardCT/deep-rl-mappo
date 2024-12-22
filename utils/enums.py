from enum import Enum

class EnvType(Enum):
    CONTINUOUS = 1
    DISCRETE = 2
    LLM = 3
    IMG = 4
    LLMv2 = 5
    
class EnvMode(Enum):
    TRAIN = 1
    TEST = 2
    VAL = 3