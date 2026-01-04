"""
Assignment 6: Alien Language Translator
Few-Shot + Chain of Thought for decoding alien messages
"""

import os
import json
from typing import List
from dataclasses import dataclass
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# ---------------- DATA CLASS ---------------- #

@dataclass
class Translation:
    alien_text: str
    human_text: str
    confidence: float
    reasoning_steps: List[str]
    cultural_notes: str


# ---------------- TRANSLATOR ---------------- #

class AlienTranslator:
    """
    AI-powered alien language translator using few-shot examples and CoT reasoning.
    """

    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.llm = ChatOpenAI(model=model_name, temperature=0.3)
        self.examples = self._load_examples()
        self._setup_chains()

    # ---------------- TODO 1 ---------------- #
    def _load_examples(self) -> List[dict]:
        """
        Example alien translations with reasoning chains.
        """

        return [
            {
                "alien": "◈◈◈ ▲▲ ●",
                "reasoning": [
                    "◈◈◈ represents quantity three",
                    "▲▲ indicates spacecraft",
                    "● marks action or state",
                ],
                "translation": "Three ships approaching",
            },
            {
                "alien": "♦♦ ◯◯ ▼",
                "reasoning": [
                    "♦♦ indicates dual entities",
                    "◯◯ represents energy or signal",
                    "▼ indicates transmission",
                ],
                "translation": "Two energy signals transmitted",
            },
            {
                "alien": "△ △ △ ■",
                "reasoning": [
                    "△ symbols represent beings",
                    "Repetition indicates plurality",
                    "■ represents location or base",
                ],
                "translation": "Three beings at the base",
            },
        ]

    # ---------------- TODO 2 ---------------- #
    def _setup_chains(self):
        example_prompt = PromptTemplate.from_template(
            """Alien Message: {alien}
Reasoning:
{reasoning}
Translation: {translation}
"""
        )

        prefix = """You are an alien linguistics expert.
You must decode alien messages using learned symbol patterns.
Think step by step and explain your reasoning clearly.

Here are known translations:
"""

        suffix = """Alien Message: {alien_message}

Step-by-step reasoning:
"""

        self.decoder_prompt = FewShotPromptTemplate(
            examples=self.examples,
            example_prompt=example_prompt,
            prefix=prefix,
            suffix=suffix,
            input_variables=["alien_message"],
        )

        self.decoder_chain = self.decoder_prompt | self.llm | StrOutputParser()

    # ---------------- TODO 3 ---------------- #
    def translate(self, alien_message: str) -> Translation:
        response = self.decoder_chain.invoke(
            {"alien_message": alien_message}
        )

        reasoning_steps = []
        translation = "Partial translation uncertain"
        confidence = 0.5
        cultural_notes = "Alien language relies heavily on symbolic repetition and spatial meaning."

        for line in response.splitlines():
            line = line.strip()
            if line.startswith("-") or line[:1].isdigit():
                reasoning_steps.append(line)
            if line.lower().startswith("translation"):
                translation = line.split(":", 1)[-1].strip()
                confidence = 0.75

        if not reasoning_steps:
            reasoning_steps = response.splitlines()

        return Translation(
            alien_text=alien_message,
            human_text=translation,
            confidence=confidence,
            reasoning_steps=reasoning_steps,
            cultural_notes=cultural_notes,
        )


# ---------------- TEST ---------------- #

def test_translator():
    translator = AlienTranslator()

    test_messages = [
        "◈◈◈◈◈ ▲▲▲ ● ◆",
        "♦♦ ◯◯◯ ▼ ★★★★",
        "△△△ ◈ ■■ ◆◆◆",
    ]

    print("👽 ALIEN LANGUAGE TRANSLATOR 👽")
    print("=" * 70)

    for msg in test_messages:
        result = translator.translate(msg)
        print(f"\nAlien: {msg}")
        print(f"Translation: {result.human_text}")
        print(f"Confidence: {result.confidence:.0%}")
        print("Reasoning:")
        for step in result.reasoning_steps[:5]:
            print(" -", step)
        print("-" * 70)


if __name__ == "__main__":
    if not os.environ.get("OPENAI_API_KEY"):
        print("⚠️ Please set OPENAI_API_KEY")
    else:
        test_translator()
