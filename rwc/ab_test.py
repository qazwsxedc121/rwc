"""Scientific A/B test framework for evaluating RWC effectiveness.

Pipeline (all LLM calls are stateless — fresh client per call, no history):
  1. Generate:  LLM creates diverse creative writing prompts
  2. Respond A:  LLM responds bare, without RWC injection
  3. Respond B:  LLM responds with RWC words as creative springboards
  4. Judge:      LLM blindly compares A vs B (order randomized), picks winner
"""

import argparse
import json
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path

from rwc import WordMixer, chinese_word_filter
from rwc.llm import chat, chat_json

# ── Prompts (all Chinese) ────────────────────────────────────────────────────

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

_BARE_SYSTEM = (
    "你是一个有创造力的作家。用中文写作。追求原创性，避免陈词滥调。"
)

_RWC_SYSTEM = (
    "你是一个有创造力的作家。用中文写作。追求原创性，避免陈词滥调。"
    "下面提供的词汇是创意跳板——它们可能帮你建立意想不到的联想，"
    "请自然地融入写作中，不要生硬罗列。"
)

_JUDGE_SYSTEM = (
    "你是一个文学评委。请客观比较两篇针对同一题目的创作。"
    "从以下三个维度评判：\n"
    "1. 语言原创性（是否避开了套话和可预测的表达）\n"
    "2. 概念新颖性（是否有出人意料的意象联结和思维跳跃）\n"
    "3. 整体感染力（是否带来情感或智识上的冲击）"
)


# ── Data types ───────────────────────────────────────────────────────────────

@dataclass
class ABConfig:
    num_rounds: int = 10
    model: str = "deepseek-v4-flash"
    judge_model: str | None = None
    rwc_distance: float = 0.5
    rwc_num_words: int = 5
    temperature: float | None = None  # None = API default (1.0)
    lang: str = "zh"  # "zh" or "en"
    seed: int | None = None

    @property
    def effective_judge_model(self) -> str:
        return self.judge_model or self.model


@dataclass
class RoundResult:
    topic: str
    prompt: str
    bare_response: str
    rwc_response: str
    rwc_words: list[str]
    winner: str  # "rwc", "bare", "tie"
    reasoning: str
    bare_is_a: bool  # True if bare response was labeled A in judging


# ── Runner ───────────────────────────────────────────────────────────────────

class ABTestRunner:
    def __init__(self, mixer: WordMixer, config: ABConfig):
        self.mixer = mixer
        self.config = config
        self._word_filter = chinese_word_filter if config.lang == "zh" else None
        if config.seed is not None:
            random.seed(config.seed)

    def run(self) -> list[RoundResult]:
        prompts = self._generate_prompts()
        results: list[RoundResult] = []
        for i, p in enumerate(prompts):
            print(f"\n{'─'*50}")
            print(f"Round {i+1}/{len(prompts)}: {p['topic']}")
            try:
                result = self._run_round(p)
                results.append(result)
                print(f"  Winner: {result.winner.upper()} — {result.reasoning}")
            except Exception as e:
                print(f"  FAILED: {e}")
        return results

    def _generate_prompts(self) -> list[dict]:
        print(f"Generating {self.config.num_rounds} prompts...")
        user = _GENERATE_USER.format(n=self.config.num_rounds)
        data = chat_json(
            user,
            system=_GENERATE_SYSTEM,
            model=self.config.model,
            max_tokens=2048,
        )
        if not isinstance(data, list):
            raise ValueError(f"Expected JSON array from generate, got: {type(data)}")
        for p in data:
            print(f"  {p.get('topic', '?')}: {p.get('prompt', '?')[:60]}...")
        return data

    def _run_round(self, prompt_info: dict) -> RoundResult:
        topic = prompt_info["topic"]
        prompt = prompt_info["prompt"]

        # Get RWC words
        rwc_words = self._get_rwc_words(topic)

        # Two conditions (order doesn't matter here, judge randomizes later)
        bare = self._respond(prompt, rwc_words=None)
        rwc_resp = self._respond(prompt, rwc_words=rwc_words)

        # Blind judge: randomly assign A/B labels
        bare_is_a = random.choice([True, False])
        if bare_is_a:
            a_text, b_text = bare, rwc_resp
        else:
            a_text, b_text = rwc_resp, bare

        winner_label, reasoning = self._judge(prompt, a_text, b_text)

        # Map label back to condition
        if winner_label == "tie":
            winner = "tie"
        elif winner_label == "A":
            winner = "bare" if bare_is_a else "rwc"
        else:
            winner = "rwc" if bare_is_a else "bare"

        return RoundResult(
            topic=topic,
            prompt=prompt,
            bare_response=bare,
            rwc_response=rwc_resp,
            rwc_words=rwc_words,
            winner=winner,
            reasoning=reasoning,
            bare_is_a=bare_is_a,
        )

    def _get_rwc_words(self, topic: str) -> list[str]:
        # Try the full topic first, then individual characters, then split on
        # common Chinese delimiters.
        candidates = [topic]
        candidates.extend(topic)  # individual characters (works for Chinese)
        candidates.extend(topic.replace("与", " ").replace("和", " ").replace("、", " ").split())

        for token in candidates:
            token = token.strip()
            if not token:
                continue
            try:
                results = self.mixer.stir(
                    token,
                    distance=self.config.rwc_distance,
                    n=self.config.rwc_num_words,
                    word_filter=self._word_filter,
                )
                words = [r.word for r in results]
                if words:
                    return words
            except Exception:
                continue
        return []

    def _respond(self, prompt: str, rwc_words: list[str] | None = None) -> str:
        if rwc_words:
            user = f"{prompt}\n\n创意跳板词汇：{'、'.join(rwc_words)}"
            system = _RWC_SYSTEM
        else:
            user = prompt
            system = _BARE_SYSTEM

        return chat(
            user,
            system=system,
            model=self.config.model,
            max_tokens=1024,
            temperature=self.config.temperature,
        )

    def _judge(
        self, prompt: str, response_a: str, response_b: str
    ) -> tuple[str, str]:
        user = (
            f"题目：{prompt}\n\n"
            f"=== 作品A ===\n{response_a}\n\n"
            f"=== 作品B ===\n{response_b}\n\n"
            '哪篇更有创造性和原创性？只返回JSON：'
            '{"winner": "A"或"B"或"tie", "reasoning": "一句话简述判决理由"}'
        )
        data = chat_json(
            user,
            system=_JUDGE_SYSTEM,
            model=self.config.effective_judge_model,
            max_tokens=512,
        )
        winner = data.get("winner", "tie")
        if winner not in ("A", "B", "tie"):
            winner = "tie"
        return winner, data.get("reasoning", "")


# ── Analysis ─────────────────────────────────────────────────────────────────

def aggregate(results: list[RoundResult]) -> dict:
    if not results:
        return {"error": "no results"}

    rwc_wins = sum(1 for r in results if r.winner == "rwc")
    bare_wins = sum(1 for r in results if r.winner == "bare")
    ties = sum(1 for r in results if r.winner == "tie")
    total = len(results)

    return {
        "total_rounds": total,
        "rwc_wins": rwc_wins,
        "bare_wins": bare_wins,
        "ties": ties,
        "rwc_win_rate": rwc_wins / total,
        "bare_win_rate": bare_wins / total,
        "tie_rate": ties / total,
        "rounds": [
            {
                "topic": r.topic,
                "prompt": r.prompt,
                "winner": r.winner,
                "reasoning": r.reasoning,
                "rwc_words": r.rwc_words,
                "bare_response": r.bare_response,
                "rwc_response": r.rwc_response,
                "bare_is_a": r.bare_is_a,
            }
            for r in results
        ],
    }


def format_report(stats: dict) -> str:
    if "error" in stats:
        return f"Error: {stats['error']}"

    n = stats["total_rounds"]
    lines = [
        "╔══════════════════════════════════════════════════════╗",
        f"║     RWC A/B 测试结果 (n={n})                         ║",
        "╠══════════════════════════════════════════════════════╣",
        f"║  RWC 胜:  {stats['rwc_wins']}/{n} ({stats['rwc_win_rate']:.0%})"
        + " " * max(0, 30 - len(f"{stats['rwc_wins']}/{n}")) + "║",
        f"║  Bare 胜: {stats['bare_wins']}/{n} ({stats['bare_win_rate']:.0%})"
        + " " * max(0, 30 - len(f"{stats['bare_wins']}/{n}")) + "║",
        f"║  平局:    {stats['ties']}/{n} ({stats['tie_rate']:.0%})"
        + " " * max(0, 30 - len(f"{stats['ties']}/{n}")) + "║",
        "╠══════════════════════════════════════════════════════╣",
    ]

    for i, r in enumerate(stats["rounds"]):
        label = {"rwc": "RWC胜", "bare": "bare胜", "tie": "平局"}[r["winner"]]
        arrow = {"rwc": "→", "bare": "←", "tie": "="}[r["winner"]]
        topic = r["topic"][:8]
        reason = r["reasoning"][:30]
        lines.append(f"║  {i+1}. {topic:<8} {arrow} {label:<6} {reason:<32}║")

    lines.append("╚══════════════════════════════════════════════════════╝")
    return "\n".join(lines)


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    # Ensure UTF-8 output on Windows terminals
    if sys.stdout.encoding != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    if sys.stderr.encoding != "utf-8":
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]

    parser = argparse.ArgumentParser(
        prog="rwc-ab-test",
        description="Scientific A/B test for RWC word injection effectiveness",
    )
    parser.add_argument(
        "--model-path", required=True, type=Path,
        help="Path to word vector file (.txt or .txt.gz)",
    )
    parser.add_argument(
        "--rounds", type=int, default=5,
        help="Number of test rounds (default: 5)",
    )
    parser.add_argument(
        "--distance", type=float, default=0.5,
        help="RWC sampling distance 0.0-1.0 (default: 0.5)",
    )
    parser.add_argument(
        "--num-words", type=int, default=5,
        help="Words to inject per round (default: 5)",
    )
    parser.add_argument(
        "--model", default="deepseek-v4-flash",
        help="LLM model for generate and respond (default: deepseek-v4-flash)",
    )
    parser.add_argument(
        "--judge-model", default=None,
        help="LLM model for judging (default: same as --model)",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--temperature", type=float, default=None,
        help="LLM temperature for responses (default: API default, usually 1.0)",
    )
    parser.add_argument(
        "--lang", default="zh", choices=["zh", "en"],
        help="Prompt language and word filter (default: zh)",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Save full results as JSON",
    )
    parser.add_argument(
        "--topic", default=None,
        help="Skip generation, use this single topic instead",
    )

    args = parser.parse_args()

    # Load word mixer
    try:
        mixer = WordMixer(str(args.model_path))
        print(f"Loaded model: {mixer.vocab_size} words, {args.model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    config = ABConfig(
        num_rounds=args.rounds,
        model=args.model,
        judge_model=args.judge_model,
        rwc_distance=args.distance,
        rwc_num_words=args.num_words,
        temperature=args.temperature,
        lang=args.lang,
        seed=args.seed,
    )

    runner = ABTestRunner(mixer, config)

    # Override generate if --topic given
    if args.topic:
        original_generate = runner._generate_prompts
        runner._generate_prompts = lambda: [
            {"topic": args.topic, "prompt": f"请以「{args.topic}」为主题，写一篇短篇散文诗或微型小说。"}
        ]
        runner._generate_prompts.__name__ = "generate_prompts_override"

    results = runner.run()

    if not results:
        print("\nNo results to report.")
        sys.exit(1)

    stats = aggregate(results)
    print("\n" + format_report(stats))

    if args.output:
        args.output.write_text(
            json.dumps(stats, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
