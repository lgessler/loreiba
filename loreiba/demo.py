import time
import conllu
from typing import Dict, List, Any

from conllu import TokenList
from tango import Step, JsonFormat
from tango.common import Tqdm


def read_conllu(path):
    with open(path, 'r') as f:
        return conllu.parse(f.read())

def sentence_to_stype_instance(sentence: TokenList) -> Dict[str, Any]:
    stype = sentence.metadata['s_type']
    tokens = [t['form'] for t in sentence]
    text = sentence.metadata['text']
    return {
        "stype": stype,
        "tokens": tokens,
        "text": text
    }

@Step.register("construct_stype_instances")
class ConstructStypeInstances(Step):
    DETERMINISTIC = True
    CACHEABLE = True
    FORMAT = JsonFormat()

    def run(self, train_conllu: str, dev_conllu: str, test_conllu: str) -> Dict[str, List[Dict[str, Any]]]:
        splits = {
            "train": [sentence_to_stype_instance(s) for s in read_conllu(train_conllu)],
            "dev": [sentence_to_stype_instance(s) for s in read_conllu(dev_conllu)],
            "test": [sentence_to_stype_instance(s) for s in read_conllu(test_conllu)]
        }
        print(splits['train'][0])
        return splits
