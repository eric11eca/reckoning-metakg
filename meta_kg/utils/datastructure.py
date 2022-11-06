from typing import Sequence, Tuple, Dict

class BiMap:
    """Maintains (bijective) mappings between two sets.
    Args:
        a (Sequence): sequence of set a elements.
        b (Sequence): sequence of set b elements.
    """

    def __init__(self, a: Sequence, b: Sequence):
        self.a_to_b = {}
        self.b_to_a = {}
        for i, j in zip(a, b):
            self.a_to_b[i] = j
            self.b_to_a[j] = i
        assert len(self.a_to_b) == len(self.b_to_a) == len(a) == len(b)

    def get_maps(self) -> Tuple[Dict, Dict]:
        """Return stored mappings.
        Returns:
            Tuple[Dict, Dict]: mappings from elements of a to b, and mappings from b to a.
        """
        return self.a_to_b, self.b_to_a

def labels_to_bimap(labels):
    """Creates mappings from label to id, and from id to label. See details in docs for BiMap.
    Args:
        labels: sequence of label to map to ids.
    Returns:
        Tuple[Dict, Dict]: mappings from labels to ids, and ids to labels.
    """
    label2id, id2label = BiMap(a=labels, b=list(range(len(labels)))).get_maps()
    return label2id, id2label