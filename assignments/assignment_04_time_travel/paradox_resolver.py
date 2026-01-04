import os
from dataclasses import dataclass
from enum import Enum
from typing import List
import re
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ---------------- ENUMS ---------------- #

class ParadoxType(Enum):
    NONE = "No Paradox"
    GRANDFATHER = "Grandfather Paradox"
    BOOTSTRAP = "Bootstrap Paradox"
    BUTTERFLY = "Butterfly Effect"
    CAUSAL_LOOP = "Causal Loop"

# ---------------- DATA CLASSES ---------------- #

@dataclass
class ParadoxAnalysis:
    scenario: str
    paradox_type: str
    reasoning_chain: List[str]
    timeline_stability: float
    resolution_strategies: List[str]
    butterfly_effects: List[str]
    final_recommendation: str

# ---------------- PARADOX RESOLVER ---------------- #

class ParadoxResolver:

    def __init__(self, model_name="gpt-4o-mini", temperature=0.3):
        self.llm = ChatOpenAI(model=model_name, temperature=temperature)
        self._setup_chains()

    def _setup_chains(self):
        # Improved Zero-shot Chain-of-Thought
        self.zero_shot_prompt = PromptTemplate.from_template(
            """
You are a time-travel paradox analyst. 

Analyze the following scenario step by step using logical deduction.
IMPORTANT: You must provide a detailed causal chain.

REQUIRED FORMAT:
1. [Step 1 of reasoning]
2. [Step 2 of reasoning]
...
Butterfly Effect: [Describe unintended consequences]
Paradox Type: [Classify it]

Scenario:
{scenario}
"""
        )

        # Few-shot examples
        examples = [
            {
                "scenario": "A traveler goes back and prevents their grandparents from meeting.",
                "analysis": """
1. Preventing the meeting stops the traveler's parents from being born.
2. If the parents are never born, the traveler is never born.
3. If the traveler is never born, they cannot go back in time to stop the meeting.
4. This creates a circular logical contradiction.
Butterfly Effect: The entire lineage of the traveler is erased from existence.
Paradox Type: Grandfather Paradox
"""
            }
        ]

        example_prompt = PromptTemplate.from_template(
            "Scenario: {scenario}\nAnalysis:\n{analysis}"
        )

        self.few_shot_prompt = FewShotPromptTemplate(
            examples=examples,
            example_prompt=example_prompt,
            prefix="Analyze the time travel scenario using step-by-step reasoning.\n\n",
            suffix="Scenario: {scenario}\nAnalysis:\n",
            input_variables=["scenario"],
        )

        self.zero_shot_chain = self.zero_shot_prompt | self.llm | StrOutputParser()
        self.few_shot_chain = self.few_shot_prompt | self.llm | StrOutputParser()

    def analyze_with_few_shot_cot(self, scenario: str) -> ParadoxAnalysis:
        response = self.few_shot_chain.invoke({"scenario": scenario})
        return self._parse_analysis(scenario, response)

    def _parse_analysis(self, scenario: str, text: str) -> ParadoxAnalysis:
        reasoning = []
        butterfly_effects = []
        
        # Improved parsing logic using Regex to find numbered steps even if they have markdown
        lines = text.splitlines()
        for line in lines:
            clean_line = line.strip()
            # Looks for lines starting with 1., 2., or **1.** etc.
            if re.match(r'^(\**\d+\.*)', clean_line):
                reasoning.append(clean_line)
            # Capture butterfly effects
            if "effect" in clean_line.lower() or "butterfly" in clean_line.lower():
                # Only add if it's not the paradox type line
                if "type:" not in clean_line.lower():
                    butterfly_effects.append(clean_line)

        # Logic for stability and strategies
        paradox_type = ParadoxType.NONE.value
        stability = 100.0
        lowered = text.lower()

        if "grandfather" in lowered:
            paradox_type = ParadoxType.GRANDFATHER.value
            stability = 15.0
            strategies = ["Prevent direct intervention", "Multiverse branching"]
        elif "bootstrap" in lowered:
            paradox_type = ParadoxType.BOOTSTRAP.value
            stability = 45.0
            strategies = ["Accept causal loop", "Introduce original source"]
        elif "butterfly" in lowered:
            paradox_type = ParadoxType.BUTTERFLY.value
            stability = 30.0
            strategies = ["Minimize interactions", "Isolate traveler"]
        else:
            strategies = ["Monitor timeline"]

        recommendation = "Timeline is unstable. Avoid interference." if stability < 50 else "Safe for limited travel."

        return ParadoxAnalysis(
            scenario=scenario,
            paradox_type=paradox_type,
            reasoning_chain=reasoning,
            timeline_stability=stability,
            resolution_strategies=strategies,
            butterfly_effects=butterfly_effects,
            final_recommendation=recommendation,
        )

def test_paradox_resolver():
    resolver = ParadoxResolver()
    scenarios = [
        "You go back in time and stop your parents from meeting.",
        "You bring back a book from the future and publish it in the past.",
        "You accidentally step on an insect in the Jurassic era.",
    ]

    for scenario in scenarios:
        analysis = resolver.analyze_with_few_shot_cot(scenario)
        print(f"\n🕰️ Scenario: {scenario}")
        print(f"Paradox Type: {analysis.paradox_type}")
        print(f"Timeline Stability: {analysis.timeline_stability}%")
        print("\nReasoning Chain:")
        for step in analysis.reasoning_chain:
            print(f"  {step}")
        print("\nButterfly Effects:")
        for effect in analysis.butterfly_effects:
            print(f"  - {effect}")
        print("\nResolution Strategies:")
        for s in analysis.resolution_strategies:
            print(f"  - {s}")
        print(f"\nFinal Recommendation: {analysis.final_recommendation}")
        print("=" * 70)

if __name__ == "__main__":
    if os.environ.get("OPENAI_API_KEY"):
        test_paradox_resolver()
    else:
        print("⚠️ Please set OPENAI_API_KEY")