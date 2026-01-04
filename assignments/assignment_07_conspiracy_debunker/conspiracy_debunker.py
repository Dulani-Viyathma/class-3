"""
Assignment 7: Conspiracy Theory Debunker
Zero-Shot + Chain of Thought for Critical Analysis
"""

import os
import json
from typing import List
from dataclasses import dataclass
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


# ---------------- DATA CLASS ---------------- #

@dataclass
class DebunkAnalysis:
    conspiracy_text: str
    main_claims: List[str]
    logical_flaws: List[str]
    reasoning_chain: List[str]
    psychological_appeal: str
    debunking_summary: str
    reliable_sources: List[str]
    confidence_score: float


# ---------------- DEBUNKER ---------------- #

class ConspiracyDebunker:
    """
    AI-powered conspiracy theory analyzer using zero-shot + CoT.
    """

    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.llm = ChatOpenAI(model=model_name, temperature=0.2)
        self._setup_chains()

    # ---------------- TODO 1 ---------------- #
    def _setup_chains(self):
        """
        Zero-shot Chain-of-Thought prompt
        """

        self.prompt = PromptTemplate.from_template(
            """
You are an expert in critical thinking and misinformation analysis.
Analyze the conspiracy theory below respectfully and logically.

Tasks:
1. Extract the main factual claims
2. Identify logical flaws or fallacies
3. Explain reasoning step by step
4. Explain why people may find it appealing
5. Debunk the theory using facts
6. Suggest reliable information sources

Respond ONLY in valid JSON.

JSON format:
{{
  "main_claims": ["..."],
  "logical_flaws": ["..."],
  "reasoning_chain": ["step 1", "step 2"],
  "psychological_appeal": "...",
  "debunking_summary": "...",
  "reliable_sources": ["..."],
  "confidence_score": 0.0
}}

Conspiracy theory:
{conspiracy_text}

Let's think step by step.
"""
        )

        self.analysis_chain = self.prompt | self.llm | StrOutputParser()

    # ---------------- TODO 2 ---------------- #
    def debunk(self, conspiracy_text: str) -> DebunkAnalysis:
        response = self.analysis_chain.invoke(
            {"conspiracy_text": conspiracy_text}
        )

        # --- SAFE JSON PARSING ---
        try:
            data = json.loads(response)
        except json.JSONDecodeError:
            data = {}

        return DebunkAnalysis(
            conspiracy_text=conspiracy_text,
            main_claims=data.get("main_claims", []),
            logical_flaws=data.get("logical_flaws", []),
            reasoning_chain=data.get("reasoning_chain", []),
            psychological_appeal=data.get("psychological_appeal", ""),
            debunking_summary=data.get("debunking_summary", ""),
            reliable_sources=data.get("reliable_sources", []),
            confidence_score=data.get("confidence_score", 0.5),
        )


# ---------------- TEST ---------------- #

def test_debunker():
    debunker = ConspiracyDebunker()

    test_theories = [
        "Birds aren't real - they're government surveillance drones. Notice how they sit on power lines to recharge?",
        "The moon landing was filmed in a Hollywood studio. The flag waves despite no atmosphere!",
        "Chemtrails from planes are mind control chemicals. Normal contrails disappear quickly but these linger!",
    ]

    print("🤔 CONSPIRACY THEORY DEBUNKER 🤔")
    print("=" * 70)

    for theory in test_theories:
        result = debunker.debunk(theory)
        print(f'\nTheory: "{theory[:60]}..."')
        print(f"Main Claims: {len(result.main_claims)} identified")
        print(f"Logical Flaws: {len(result.logical_flaws)} found")
        print(f"Confidence: {result.confidence_score:.0%}")
        print("-" * 70)


if __name__ == "__main__":
    if not os.environ.get("OPENAI_API_KEY"):
        print("⚠️ Please set OPENAI_API_KEY")
    else:
        test_debunker()
