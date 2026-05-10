"""Exquisite Corpse phrase constructor — builds structured surreal phrases
from POS-tagged word vector vocabulary for creative injection."""

import random
import re
from collections import defaultdict

import jieba.posseg as pseg

from rwc.engine import WordVectorEngine

# Jieba tags
NOUN_TAGS = {"n", "nz", "ng"}           # 名词/其他专名/名语素
NAME_TAGS = {"nr", "ns", "nt"}           # 人名/地名/机构（单独放，谨慎用）
VERB_TAGS = {"v", "vd", "vn"}            # 动词
ADJ_TAGS  = {"a", "ad", "an", "ag"}     # 形容词
ALL_TAGS  = NOUN_TAGS | NAME_TAGS | VERB_TAGS | ADJ_TAGS


def _is_clean(word: str) -> bool:
    """Reject words with Latin letters, digits, symbols, or excessive length."""
    if len(word) < 1 or len(word) > 3:
        return False
    if re.search(r"[a-zA-Z0-9]", word):
        return False
    if re.search(r"[<>\(\)\[\]\{\}@://\.\,\-\+\/\\\%\s]", word):
        return False
    return True


def tag_vocabulary(engine: WordVectorEngine) -> dict[str, set[str]]:
    """Tag every word with jieba POS. Only keeps clean, short words."""
    bank: dict[str, set[str]] = defaultdict(set)
    vocab = engine.vocabulary()
    n = len(vocab)
    for i, word in enumerate(vocab):
        if not _is_clean(word):
            continue
        if i % 20000 == 0:
            print(f"  tagging {i}/{n}...")
        for _, flag in pseg.cut(word):
            if flag in ALL_TAGS:
                bank[flag].add(word)
            break  # first token only
    return dict(bank)


class PhraseMixer:
    """Generates structured surreal phrases from POS-tagged vocabulary.

    Uses word vectors for semantic filtering — instead of pure random,
    picks nouns that have moderate similarity to the topic word.
    """

    def __init__(self, engine: WordVectorEngine):
        self.engine = engine
        bank = tag_vocabulary(engine)

        self.nouns   = sorted(bank.get("n", set()) | bank.get("nz", set()) | bank.get("ng", set()))
        self.verbs   = sorted(bank.get("v", set()) | bank.get("vd", set()) | bank.get("vn", set()))
        self.adjs    = sorted(bank.get("a", set()) | bank.get("ad", set()) | bank.get("an", set()) | bank.get("ag", set()))
        self.persons = sorted(bank.get("nr", set()))
        self.places  = sorted(bank.get("ns", set()))
        self.orgs    = sorted(bank.get("nt", set()))

        # Build similarity index for fast semantic lookup
        self._nouns_list = self.nouns
        self._noun_indices = [self.engine.word_to_index[w]
                              for w in self._nouns_list
                              if w in self.engine.word_to_index]

        print(f"Clean vocab: {len(self.nouns)} nouns, {len(self.verbs)} verbs, "
              f"{len(self.adjs)} adjs, {len(self.persons)} persons, "
              f"{len(self.places)} places")

    def _semantic_nouns(self, topic: str, n: int = 50,
                        distance: float = 0.3) -> list[str]:
        """Find nouns with moderate semantic distance to the topic."""
        if topic not in self.engine.word_to_index:
            return random.sample(self.nouns, min(n, len(self.nouns)))

        sims = self.engine.similarities(topic)
        topic_idx = self.engine.word_to_index[topic]

        # Score each indexed noun by similarity
        scored = []
        for idx in self._noun_indices:
            if idx == topic_idx:
                continue
            scored.append((self.engine._index_to_word[idx], float(sims[idx])))
        scored.sort(key=lambda x: x[1], reverse=True)

        # Map distance to percentile range
        center = distance ** 2
        lo = int(center * len(scored))
        hi = int((center + 0.15) * len(scored))
        hi = max(lo + n, hi)

        candidates = scored[lo:hi]
        random.shuffle(candidates)
        return [w for w, _ in candidates[:n]]

    def surprise(self, topic: str | None = None,
                 template: str = "nvn",
                 semantic: bool = True) -> str:
        """Generate one surreal phrase.

        Templates:
          nvn   = topic + verb + noun        (default, most reliable)
          nvnr  = topic + verb + person_name
          nvns  = topic + verb + place_name
          avn   = adjective + verb + noun
          nrnv  = person_name + verb + noun
          nn    = noun + noun                (compound surrealism)
        """
        if topic is None:
            topic = random.choice(self.nouns)

        v = random.choice(self.verbs)

        if template == "nvn":
            if semantic and topic in self.engine.word_to_index:
                pool = self._semantic_nouns(topic, n=50, distance=0.3)
                n2 = random.choice(pool) if pool else random.choice(self.nouns)
            else:
                n2 = random.choice(self.nouns)
            return f"{topic}{v}{n2}"

        elif template == "nvnr":
            n2 = random.choice(self.persons) if self.persons else random.choice(self.nouns)
            return f"{topic}{v}{n2}"

        elif template == "nvns":
            n2 = random.choice(self.places) if self.places else random.choice(self.nouns)
            return f"{topic}{v}{n2}"

        elif template == "avn":
            a = random.choice(self.adjs)
            if semantic and topic in self.engine.word_to_index:
                pool = self._semantic_nouns(topic, n=30, distance=0.3)
                n2 = random.choice(pool) if pool else random.choice(self.nouns)
            else:
                n2 = random.choice(self.nouns)
            return f"{a}{v}{n2}"

        elif template == "nrnv":
            n1 = random.choice(self.persons) if self.persons else random.choice(self.nouns)
            n2 = random.choice(self.nouns)
            return f"{n1}{v}{n2}"

        elif template == "nn":
            if semantic and topic in self.engine.word_to_index:
                pool = self._semantic_nouns(topic, n=50, distance=0.3)
                n2 = random.choice(pool) if pool else random.choice(self.nouns)
            else:
                n2 = random.choice(self.nouns)
            return f"{topic}{n2}"

        else:
            return self.surprise(topic=topic, semantic=semantic)

    def surprises(self, topic: str, n: int = 5,
                  templates: list[str] | None = None,
                  semantic: bool = True) -> list[str]:
        if templates is None:
            templates = ["nvn", "nvnr", "nvns", "nn"]
        results = []
        for _ in range(n):
            tpl = random.choice(templates)
            results.append(self.surprise(topic=topic, template=tpl, semantic=semantic))
        return results

    def stats(self) -> dict:
        return {
            "nouns": len(self.nouns), "verbs": len(self.verbs),
            "adjectives": len(self.adjs), "persons": len(self.persons),
            "places": len(self.places), "organizations": len(self.orgs),
        }
