import torch
import bert_score
import torch.nn as nn
from typing import Dict, List, Set, Optional
import re
import unicodedata

class ClinicalReward:
    def __init__(self):
        self.anatomy_keywords = {
            'intervertebral', 'vertebral', 'disc', 'lumbar', 'spinal', 'canal',
            'vertebrae', 'body', 'sac', 'dural', 'nerve', 'cord', 'root', 'endplate',
            'ligament', 'facet', 'foramen', 'recess', 'spinous', 'fossa', 'vertebra',
            'bodies', 'roots', 'endplates', 'ligaments', 'foramina', 'recesses', 'discs',
            'sacral', 'foramen', 'endplates', 'spines', 'ligamentum', 'annular', 'pulposus',
            'nucleus'
        }

        self.pathology_keywords = {
            'bulging', 'protrusion', 'herniation', 'compression', 'degeneration',
            'stenosis', 'narrowing', 'inflammation', 'hyperplasia', 'spondylolisthesis',
            'thickening', 'fracture', 'cyst', 'edema', 'degenerative', 'hyperosteogeny',
            'protruded', 'bulged', 'herniated', 'compressed', 'narrowed', 'thickened',
            'bulges', 'protrudes', 'prolapse', 'prolapsed', 'spondylosis', 'schmorl\'',
            'node', 'nodes', 'nodules', 'nodular', 'lesion', 'lesions', 'nodule', 'scoliosis',
            'cysts', 'fractures', 'compressing', 'protruding', 'narrower', 'narrows',
            'flavum', 'hyperostosis', 'spondylosis', 'spondylolysis', 'spondylitis'
        }

        self.severity_keywords = {
            'mild', 'slight', 'moderate', 'severe', 'significant', 'obvious', 'evident',
            'reduced', 'decreased', 'compressed', 'narrowed', 'patchy', 'multiple',
            'slightly', 'obviously', 'significantly', 'mildly'
        }

        self.location_keywords = {
            'anterior', 'posterior', 'central', 'lateral', 'paracentral', 'posterolateral',
            'left', 'right', 'bilateral', 'superior', 'inferior', 'posteriorly', 'anteriorly',
            'anteroposterior', 'anterior-posterior'
        }

        self.position_entities = {
            'l1/2', 'l2/3', 'l3/4', 'l4/5', 'l5/s1', 'l1/l2', 'l2/l3', 'l3/l4', 'l4/l5',
            'l1', 'l2', 'l3', 'l4', 'l5', 's1', 't12', 't11', 's2', 's3', 's4', 't10',
            'l2/3-l5/s1','l1/2-l5/s1', 'l3/4-l5/s1', 'l3-s1', 'l4-s1', 'l5-s1', 't12/l1',
            'l3-4', 'l2-s1', 'l1/2-l4/5', 'l3-5', 'l1-3', 'l3/l4-l5/s1', 'l2/l3-l5/s1',
            't11/t12', 'l2-l4', 'l1-4', 'l1-l4', 'l2-l5', 'l1-l3', 't11-l3', 'l2/l3-l4/l5',
            'l1-l5', 't10/11', 't11-12', 's1-2', 'l4-5', 'l2-3', 'l3-l5', 'l1-5', 'l2-4'
        }

        self.lemmatization_map = {
            'discs': 'disc', 'bodies': 'body', 'roots': 'root', 'endplates': 'endplate',
            'ligaments': 'ligament', 'foramina': 'foramen', 'recesses': 'recess',
            'vertebrae': 'vertebra', 'nodules': 'nodule', 'lesions': 'lesion',
            'cysts': 'cyst', 'fractures': 'fracture', 'spines': 'spine', 'nodes': 'node',

            'protruded': 'protrusion', 'bulged': 'bulging', 'herniated': 'herniation',
            'compressed': 'compression', 'narrowed': 'narrow', 'thickened': 'thickening',
            'displaced': 'displacement', 'compressing': 'compression', 'protruding': 'protrusion',
            'bulges': 'bulging', 'protrudes': 'protrusion', 'narrows': 'narrow',
            'flattens': 'flattening', 'straightens': 'straightening', 'narrower': 'narrow',
            'prolapsed': 'prolapse', 'narrowing': 'narrow',

            'slightly': 'slight', 'obviously': 'obvious', 'significantly': 'significant',
            'mildly': 'mild', 'degenerative': 'degeneration', 'posteriorly': 'posterior',
            'anteriorly': 'anterior',
        }

        self.synonym_map = {
            'l2/3-l5/s1': ('l2/3', 'l3/4', 'l4/5', 'l5/s1'),
            'l1/2-l5/s1': ('l1/2', 'l2/3', 'l3/4', 'l4/5', 'l5/s1'),
            'l3/4-l5/s1': ('l3/4', 'l4/5', 'l5/s1'),
            'l3-s1': ('l3', 'l4', 'l5', 's1'),
            'l4-s1': ('l4', 'l5', 's1'),
            'l5-s1': 'l5/s1',
            'l3-4': 'l3/4',
            'l2-s1': ('l2', 'l3', 'l4', 'l5', 's1'),
            'l1/2-l4/5': ('l1/2', 'l2/3', 'l3/4', 'l4/5'),
            'l2/l3-l5/s1': ('l2/3', 'l3/4', 'l4/5', 'l5/s1'),
            'l2-l4': ('l2', 'l3', 'l4'),
            'l1-4': ('l1', 'l2', 'l3', 'l4'),
            'l2-l4': ('l2', 'l3', 'l4'),
            'l2-l5': ('l2', 'l3', 'l4', 'l5'),
            'l1-l3': ('l1', 'l2', 'l3'),
            't11-l3': ('t11', 't12', 'l1', 'l2', 'l3'),
            'l2/l3-l4/l5': ('l2/l3', 'l3/l4', 'l4/l5'),
            'l1-l5': ('l1', 'l2', 'l3', 'l4', 'l5'),
            's1-2': 's1/s2',
            't11-12': 't11/t12',
            'l4-5': 'l4/5',
            'l2-3': 'l2/3',
            'l1-2': 'l1/2',
            'l3-l5': ('l3', 'l4', 'l5'),
            'l1-5': ('l1', 'l2', 'l3', 'l4', 'l5'),
            'l2-4': ('l2', 'l3', 'l4'),
            'l1/l2': 'l1/2',
            'l2/l3': 'l2/3',
            'l3/l4': 'l3/4',
            'l4/l5': 'l4/5',

            'intervertebral': 'disc', 'spinal': 'cord', 'dural': 'sac',
            'ligamentum': 'ligament', 'flavum': 'ligament', 'annular': 'disc',
            'nucleus': 'disc', 'pulposus': 'disc',

            'narrow': 'stenosis', 'hyperosteogeny': 'hyperplasia',
            'hyperostosis': 'hyperplasia', 'spondylitis': 'spondylolisthesis',
            'spondylosis': 'spondylolisthesis', 'spondylolysis': 'spondylolisthesis',
            'spondylitis': 'spondylolisthesis', 'anterior-posterior': 'anteroposterior'
        }

        self.entity_weights = {
            'position': 2.0,
            'pathology': 1.5,
            'severity': 1.2,
            'anatomy': 1.0,
            'location': 0.8,
        }

    def custom_tokenizer(self, text: str):
        def to_ascii(token: str):
            normalized = unicodedata.normalize('NFKD', token)
            ascii_token = normalized.encode('ascii', 'ignore').decode('ascii')
            return ascii_token
        tokens = text.split()
        processed_tokens = []
        for token in tokens:
            token_lower = token.lower()
            if '/' in token_lower or '-' in token_lower:
                parts = re.split(r'([^\w/_-])', token_lower)
                parts = [p for p in parts if p.strip()]
                processed_tokens.extend(parts)
            else:
                parts = re.split(r'(\W)', token_lower)
                parts = [p for p in parts if p.strip()]
                processed_tokens.extend(parts)
        ascii_tokens = []
        for token in processed_tokens:
            ascii_token = to_ascii(token)
            if ascii_token.strip():
                ascii_tokens.append(ascii_token)
        return ascii_tokens

    def normalize_tokens(self, keywords: Set[str]) -> Set[str]:
        normalized = set()
        for word in keywords:
            lemma = self.lemmatization_map.get(word, word)
            normalized_word = self.synonym_map.get(lemma, lemma)
            if isinstance(normalized_word, str):
                normalized.add(normalized_word)
            elif isinstance(normalized_word, tuple):
                normalized.update(normalized_word)
        return normalized

    def get_classified_tokens(self, report: str) -> Dict[str, Set[str]]:
        entities = self.extract_clinical_entities(report)
        new_entities = {}
        for key, value in entities.items():
            new_entities[key] = self.normalize_tokens(value)
        return new_entities

    def extract_clinical_entities(self, text: str) -> Dict[str, Set[str]]:
        tokens = self.custom_tokenizer(text)
        entities = {
            'position': set(),
            'pathology': set(),
            'severity': set(),
            'anatomy': set(),
            'location': set()
        }

        for token in tokens:
            if token in self.position_entities:
                entities['position'].add(token)
            elif token in self.pathology_keywords:
                entities['pathology'].add(token)
            elif token in self.severity_keywords:
                entities['severity'].add(token)
            elif token in self.anatomy_keywords:
                entities['anatomy'].add(token)
            elif token in self.location_keywords:
                entities['location'].add(token)
        return entities

    def calculate_F1(self, true_entities: Optional[List[str] | Set[str]], predicted_entities: Optional[List[str] | Set[str]]):
        if isinstance(true_entities, list):
            true_entities = set(true_entities)
        if isinstance(predicted_entities, list):
            predicted_entities = set(predicted_entities)

        true_positives = len(true_entities & predicted_entities)
        false_positives = len(predicted_entities - true_entities)
        false_negatives = len(true_entities - predicted_entities)

        precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0.0
        return f1

    def calculate_weighted_f1(self, ref_entities: Dict[str, Set[str]],
                            pred_entities: Dict[str, Set[str]]) -> float:
        weighted_f1 = 0.0
        total_weights = sum(list(self.entity_weights.values()))
        for entity_type in self.entity_weights.keys():
            f1 = self.calculate_F1(ref_entities[entity_type], pred_entities[entity_type])
            weighted_f1 += f1 * self.entity_weights[entity_type] / total_weights
        return weighted_f1

    def clinical_reward(self, reference_report: str, predicted_report: str) -> float:
        ref_entities = self.get_classified_tokens(reference_report)
        pred_entities = self.get_classified_tokens(predicted_report)
        weighted_f1 = self.calculate_weighted_f1(ref_entities, pred_entities)
        return weighted_f1

class BertScorer(nn.Module):
    def __init__(self, batch_size):
        super().__init__()
        with torch.no_grad():
            self.mybert_scorer = bert_score.BERTScorer(
                                        num_layers=5,
                                        batch_size=batch_size,
                                        all_layers=False,
                                        idf=False,
                                        device='cuda',
                                        lang='en',
                                        rescale_with_baseline=True,     # rescale the score to [0, 1]
                                        baseline_path=None,
                                        use_fast_tokenizer=True
                                        )

    def forward(self, refs, preds):
        p, r, f = self.mybert_scorer.score(preds, refs)
        # f.shape torch.size([B])
        return f.cpu().numpy().tolist()

if __name__ == "__main__":
    bert_scorer = BertScorer(batch_size=1)
    clinical_reward = ClinicalReward()
