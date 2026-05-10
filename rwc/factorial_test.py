"""2×2 factorial A/B test: Temperature × RWC.

Compares four conditions per round:
  - Bare + Low temp  (temperature 0.3, no RWC)
  - RWC  + Low temp  (temperature 0.3, RWC injection)
  - Bare + High temp (temperature 1.0, no RWC)
  - RWC  + High temp (temperature 1.0, RWC injection)

Judge ranks all four blindly (labels randomized), best to worst.
"""

import argparse
import json
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path

from rwc import WordMixer, chinese_word_filter
from rwc.llm import chat, chat_json

# ── Prompts ──────────────────────────────────────────────────────────────────

_GENERATE_SYSTEM = (
    "你是一个创意写作命题专家。生成多样化的短篇散文诗或微型小说题目（100-300字），"
    "主题跨越多领域：自然、科技、哲学、日常生活、超现实、社会观察等。"
    "题目要具体、有画面感，避免过于抽象和空泛。"
)

_GENERATE_USER = (
    "生成{n}个创意写作题目。topic必须是单个中文词汇或字（如：荒原、时间、火、海），"
    "不要用短语或多个词组合。"
    '只返回JSON数组：[{{"topic": "单个词", "prompt": "具体的写作提示..."}}]'
)

_CREATIVE_SYSTEM = (
    "你是一个有创造力的作家。用中文写作。"
    "调动具体的感官记忆——气味、触感、温度、声音。"
    "将日常之物与宏大之物并置，制造张力。"
    "追求原创性，避免陈词滥调。"
)

_RWC_SYSTEM = (
    "你是一个有创造力的作家。用中文写作。"
    "调动具体的感官记忆——气味、触感、温度、声音。"
    "下面提供的词汇是创意跳板——它们可能帮你建立意想不到的联想，"
    "请自然地融入写作中，不要生硬罗列。"
    "追求原创性，避免陈词滥调。"
)

_JUDGE_SYSTEM = (
    "你是一个文学评委。请比较四篇作品，从最好到最差排序。\n"
    "从以下五个维度综合评判：\n"
    "1. 语言的精准与鲜活\n"
    "2. 意象的连贯与自洽\n"
    "3. 结构的完整与节奏\n"
    "4. 情感的深度与余味\n"
    "5. 思想的锋芒与分量\n"
    "不要偏向奇诡或平实任何一种风格，只判断哪篇在整体上更好。"
)


# ── Data types ───────────────────────────────────────────────────────────────

@dataclass
class FactorialConfig:
    num_rounds: int = 10
    model: str = "deepseek-v4-flash"
    judge_model: str | None = None
    rwc_distance: float = 0.5
    rwc_num_words: int = 5
    temp_low: float = 0.3
    temp_high: float = 1.0
    seed: int | None = None

    @property
    def effective_judge_model(self) -> str:
        return self.judge_model or self.model


CONDITION_NAMES = ["bare_low", "rwc_low", "bare_high", "rwc_high"]
CONDITION_LABELS = {
    "bare_low": "Bare+T=0.3",
    "rwc_low": "RWC+T=0.3",
    "bare_high": "Bare+T=1.0",
    "rwc_high": "RWC+T=1.0",
}


@dataclass
class FactorialRoundResult:
    topic: str
    prompt: str
    rwc_words: list[str]
    responses: dict[str, str]  # condition_name -> text
    ranking: list[str]         # condition names, best first
    reasoning: str


# ── Runner ───────────────────────────────────────────────────────────────────

class FactorialRunner:
    def __init__(self, mixer: WordMixer, config: FactorialConfig):
        self.mixer = mixer
        self.config = config
        if config.seed is not None:
            random.seed(config.seed)

    def run(self) -> list[FactorialRoundResult]:
        prompts = self._generate_prompts()
        results: list[FactorialRoundResult] = []
        for i, p in enumerate(prompts):
            print(f"\n{'─'*50}")
            print(f"Round {i+1}/{len(prompts)}: {p['topic']}")
            try:
                result = self._run_round(p)
                results.append(result)
                top = result.ranking[0]
                print(f"  1st: {CONDITION_LABELS[top]}")
                print(f"  2nd: {CONDITION_LABELS[result.ranking[1]]}")
                print(f"  3rd: {CONDITION_LABELS[result.ranking[2]]}")
                print(f"  4th: {CONDITION_LABELS[result.ranking[3]]}")
            except Exception as e:
                print(f"  FAILED: {e}")
        return results

    def _generate_prompts(self) -> list[dict]:
        print(f"Generating {self.config.num_rounds} prompts...")
        user = _GENERATE_USER.format(n=self.config.num_rounds)
        data = chat_json(user, system=_GENERATE_SYSTEM, model=self.config.model, max_tokens=2048)
        if not isinstance(data, list):
            raise ValueError(f"Expected JSON array, got: {type(data)}")
        for p in data:
            print(f"  {p.get('topic', '?')}: {p.get('prompt', '?')[:60]}...")
        return data

    def _run_round(self, prompt_info: dict) -> FactorialRoundResult:
        topic = prompt_info["topic"]
        prompt = prompt_info["prompt"]
        rwc_words = self._get_rwc_words(topic)

        # Generate 4 responses (2×2)
        responses = {
            "bare_low": self._respond(prompt, rwc_words=None, temperature=self.config.temp_low),
            "rwc_low": self._respond(prompt, rwc_words=rwc_words, temperature=self.config.temp_low),
            "bare_high": self._respond(prompt, rwc_words=None, temperature=self.config.temp_high),
            "rwc_high": self._respond(prompt, rwc_words=rwc_words, temperature=self.config.temp_high),
        }

        # Blind ranking: randomize labels A-D
        items = list(responses.items())  # [(name, text), ...]
        random.shuffle(items)
        label_map = {}
        for i, (name, _) in enumerate(items):
            label_map[chr(65 + i)] = name  # A->rwc_low, B->bare_high, etc.

        ranking_labels, reasoning = self._judge_rank(prompt, [(chr(65 + i), text) for i, (_, text) in enumerate(items)])
        ranking = [label_map[label] for label in ranking_labels]

        return FactorialRoundResult(
            topic=topic,
            prompt=prompt,
            rwc_words=rwc_words,
            responses=responses,
            ranking=ranking,
            reasoning=reasoning,
        )

    def _get_rwc_words(self, topic: str) -> list[str]:
        candidates = [topic]
        candidates.extend(topic)
        candidates.extend(topic.replace("与", " ").replace("和", " ").replace("、", " ").split())
        for token in candidates:
            token = token.strip()
            if not token:
                continue
            try:
                results = self.mixer.stir(token, distance=self.config.rwc_distance, n=self.config.rwc_num_words, word_filter=chinese_word_filter)
                words = [r.word for r in results]
                if words:
                    return words
            except Exception:
                continue
        return []

    def _respond(self, prompt: str, rwc_words: list[str] | None, temperature: float) -> str:
        if rwc_words:
            user = f"{prompt}\n\n创意跳板词汇：{'、'.join(rwc_words)}"
            system = _RWC_SYSTEM
        else:
            user = prompt
            system = _CREATIVE_SYSTEM
        return chat(user, system=system, model=self.config.model, max_tokens=1024, temperature=temperature)

    def _judge_rank(self, prompt: str, labeled_items: list[tuple[str, str]]) -> tuple[list[str], str]:
        parts = [f"题目：{prompt}\n"]
        for label, text in labeled_items:
            parts.append(f"=== 作品{label} ===\n{text}\n")
        parts.append(
            '从最好到最差排序（1st=最好的文学作品, 4th=最差的）。'
            '只返回JSON：{"ranking": ["A","B","C","D"], "reasoning": "一句话简述排序理由"}'
        )
        data = chat_json(
            "\n".join(parts),
            system=_JUDGE_SYSTEM,
            model=self.config.effective_judge_model,
            max_tokens=512,
            temperature=0.3,
        )
        ranking = data.get("ranking", [])
        valid = [l for l in ranking if l in ("A", "B", "C", "D")]
        if len(valid) != 4:
            valid = ["A", "B", "C", "D"]
        return valid, data.get("reasoning", "")


# ── Analysis ─────────────────────────────────────────────────────────────────

def aggregate(results: list[FactorialRoundResult]) -> dict:
    if not results:
        return {"error": "no results"}

    # Count 1st place wins per condition
    wins = {name: 0 for name in CONDITION_NAMES}
    rank_scores = {name: 0 for name in CONDITION_NAMES}  # 4=1st, 3=2nd, 2=3rd, 1=4th
    points_map = {0: 4, 1: 3, 2: 2, 3: 1}

    for r in results:
        top = r.ranking[0]
        wins[top] += 1
        for rank_idx, name in enumerate(r.ranking):
            rank_scores[name] += points_map[rank_idx]

    total = len(results)
    max_score = total * 4  # if always ranked 1st

    return {
        "total_rounds": total,
        "temp_low": results[0].responses and 0.3,  # from config
        "temp_high": 1.0,
        "first_place_wins": wins,
        "first_place_rates": {n: wins[n] / total for n in CONDITION_NAMES},
        "avg_rank_scores": {n: rank_scores[n] / total for n in CONDITION_NAMES},
        "rank_score_pct": {n: rank_scores[n] / max_score * 100 for n in CONDITION_NAMES},
        "rounds": [
            {
                "topic": r.topic,
                "prompt": r.prompt,
                "rwc_words": r.rwc_words,
                "ranking": r.ranking,
                "reasoning": r.reasoning,
                "responses": r.responses,
            }
            for r in results
        ],
    }


def format_report(stats: dict) -> str:
    if "error" in stats:
        return f"Error: {stats['error']}"

    n = stats["total_rounds"]
    wins = stats["first_place_wins"]
    rates = stats["first_place_rates"]
    scores = stats["avg_rank_scores"]

    lines = [
        "╔════════════════════════════════════════════════════════════╗",
        "║     2×2 因子实验：Temperature × RWC  (n={:2d})            ║".format(n),
        "╠════════════════════════════════════════════════════════════╣",
        "║  条件              │ 1st次数 │ 1st率  │ 平均排名分       ║",
        "╠════════════════════════════════════════════════════════════╣",
    ]

    for name in CONDITION_NAMES:
        label = CONDITION_LABELS[name]
        star = " ★" if rates[name] == max(rates.values()) and rates[name] > 0 else ""
        lines.append(
            f"║  {label:<18} │ {wins[name]:>3}/{n}   │ {rates[name]:>5.0%}  │ {scores[name]:.2f}/4.0{'':<10}║"
        )

    lines.append("╠════════════════════════════════════════════════════════════╣")
    lines.append("║  各轮第一名:                                              ║")
    for i, r in enumerate(stats["rounds"]):
        top = r["ranking"][0]
        lines.append(f"║  {i+1:2d}. {r['topic']:<8} → {CONDITION_LABELS[top]:<20}                      ║")
    lines.append("╚════════════════════════════════════════════════════════════╝")
    return "\n".join(lines)


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    if sys.stdout.encoding != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if sys.stderr.encoding != "utf-8":
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        prog="rwc-factorial",
        description="2×2 factorial test: Temperature × RWC",
    )
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--distance", type=float, default=0.5)
    parser.add_argument("--num-words", type=int, default=5)
    parser.add_argument("--temp-low", type=float, default=0.3)
    parser.add_argument("--temp-high", type=float, default=1.0)
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--judge-model", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--topic", default=None)

    args = parser.parse_args()

    try:
        mixer = WordMixer(str(args.model_path))
        print(f"Loaded model: {mixer.vocab_size} words, {args.model_path}")
        print(f"T_low={args.temp_low}  T_high={args.temp_high}  RWC d={args.distance}")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    config = FactorialConfig(
        num_rounds=args.rounds,
        model=args.model,
        judge_model=args.judge_model,
        rwc_distance=args.distance,
        rwc_num_words=args.num_words,
        temp_low=args.temp_low,
        temp_high=args.temp_high,
        seed=args.seed,
    )

    runner = FactorialRunner(mixer, config)

    if args.topic:
        runner._generate_prompts = lambda: [
            {"topic": args.topic, "prompt": f"请以「{args.topic}」为主题，写一篇短篇散文诗或微型小说。"}
        ]

    results = runner.run()

    if not results:
        print("\nNo results.")
        sys.exit(1)

    stats = aggregate(results)
    print("\n" + format_report(stats))

    if args.output:
        args.output.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
