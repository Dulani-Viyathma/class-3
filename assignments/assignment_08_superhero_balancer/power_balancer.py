"""
Assignment 8: Superhero Power Balancer
All Prompting Techniques Combined
"""

import os
import json
from typing import Dict, List
from dataclasses import dataclass
from enum import Enum
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# ---------------- ENUMS ---------------- #

class PowerType(Enum):
    PHYSICAL = "physical"
    ENERGY = "energy"
    MENTAL = "mental"
    REALITY = "reality"
    TECH = "technology"
    MAGIC = "magic"


# ---------------- DATA CLASSES ---------------- #

@dataclass
class Hero:
    name: str
    abilities: List[str]
    power_type: str
    power_level: float
    weaknesses: List[str]
    synergies: List[str]


@dataclass
class BalanceReport:
    hero: Hero
    analysis_method: str
    power_rating: float
    balance_issues: List[str]
    suggested_changes: List[str]
    team_synergies: Dict[str, float]
    counter_picks: List[str]


# ---------------- BALANCER ---------------- #

class PowerBalancer:
    """
    AI-powered game balancer using zero-shot, few-shot, and Chain-of-Thought.
    """

    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.llm = ChatOpenAI(model=model_name, temperature=0.4)
        self._setup_chains()

    # ---------------- SETUP CHAINS ---------------- #

    def _setup_chains(self):

        # ZERO-SHOT: Ability analysis
        self.ability_chain = (
            PromptTemplate.from_template(
                """
Analyze the following superhero ability for a competitive fighting game.

Tasks:
- Estimate power level (0–10)
- Identify potential exploits
- Identify counter-play options

Ability:
{ability}

Respond ONLY in JSON:
{{
  "power": 0.0,
  "exploits": ["..."],
  "counters": ["..."]
}}
"""
            )
            | self.llm
            | StrOutputParser()
        )

        # FEW-SHOT: Power classification
        examples = [
            {
                "ability": "Super strength and invulnerability",
                "type": "physical",
                "reasoning": "Enhances raw physical power",
            },
            {
                "ability": "Telepathy and mind control",
                "type": "mental",
                "reasoning": "Manipulates thoughts and perception",
            },
            {
                "ability": "Time manipulation",
                "type": "reality",
                "reasoning": "Alters fundamental rules of reality",
            },
            {
                "ability": "Energy beams and force fields",
                "type": "energy",
                "reasoning": "Projects and controls energy",
            },
        ]

        self.type_chain = (
            FewShotPromptTemplate(
                examples=examples,
                example_prompt=PromptTemplate.from_template(
                    "Ability: {ability}\nType: {type}\nReasoning: {reasoning}\n"
                ),
                prefix="Classify the superhero ability type.\n\n",
                suffix="Ability: {ability}\nType:",
                input_variables=["ability"],
            )
            | self.llm
            | StrOutputParser()
        )

        # CoT: Interaction calculation
        self.interaction_chain = (
            PromptTemplate.from_template(
                """
Analyze how these two heroes interact in combat.

Hero 1 abilities:
{a1}

Hero 2 abilities:
{a2}

Let's think step by step.
Finally, output ONLY a synergy score between 0 and 1.
"""
            )
            | self.llm
            | StrOutputParser()
        )

    # ---------------- ZERO-SHOT ANALYSIS ---------------- #

    def analyze_hero_zero_shot(self, hero: Hero) -> Dict[str, any]:
        total_power = 0.0
        exploits = []
        counters = []

        for ability in hero.abilities:
            response = self.ability_chain.invoke({"ability": ability})

            # 🔧 FIX: strip code fences
            cleaned = response.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.replace("```json", "").replace("```", "").strip()

            try:
                data = json.loads(cleaned)
                total_power += data.get("power", 0)
                exploits.extend(data.get("exploits", []))
                counters.extend(data.get("counters", []))
            except json.JSONDecodeError:
                continue

        power_level = min(10.0, total_power / max(1, len(hero.abilities)))

        return {
            "power_level": power_level,
            "exploits": list(set(exploits)),
            "counters": list(set(counters)),
        }

    # ---------------- FEW-SHOT CLASSIFICATION ---------------- #

    def classify_power_few_shot(self, abilities: List[str]) -> str:
        votes = []

        for ability in abilities:
            result = self.type_chain.invoke({"ability": ability}).lower()
            for pt in PowerType:
                if pt.value in result:
                    votes.append(pt.value)

        return max(set(votes), key=votes.count) if votes else PowerType.PHYSICAL.value

    # ---------------- CoT SYNERGY ---------------- #

    def calculate_synergy_cot(self, hero1: Hero, hero2: Hero) -> float:
        response = self.interaction_chain.invoke(
            {"a1": ", ".join(hero1.abilities), "a2": ", ".join(hero2.abilities)}
        )

        for token in response.split():
            try:
                value = float(token)
                if 0 <= value <= 1:
                    return value
            except ValueError:
                continue

        return 0.5

    # ---------------- COMBINED BALANCE ---------------- #

    def detect_imbalance_combined(self, hero: Hero, meta: List[Hero]) -> BalanceReport:
        analysis = self.analyze_hero_zero_shot(hero)
        power_type = self.classify_power_few_shot(hero.abilities)

        synergies = {
            other.name: self.calculate_synergy_cot(hero, other)
            for other in meta
        }

        balance_issues = []
        suggestions = []

        if analysis["power_level"] > 8:
            balance_issues.append("Overpowered abilities")
            suggestions.append("Increase cooldowns or add resource costs")

        if power_type == PowerType.REALITY.value:
            balance_issues.append("Reality manipulation breaks game rules")
            suggestions.append("Limit duration or add strict counters")

        return BalanceReport(
            hero=hero,
            analysis_method="combined",
            power_rating=analysis["power_level"],
            balance_issues=balance_issues,
            suggested_changes=suggestions,
            team_synergies=synergies,
            counter_picks=analysis["counters"][:3],
        )

    # ---------------- BONUS ---------------- #

    def auto_balance(self, hero: Hero, target_power: float) -> Hero:
        analysis = self.analyze_hero_zero_shot(hero)

        if analysis["power_level"] > target_power:
            hero.weaknesses.append("Limited stamina")
        elif analysis["power_level"] < target_power:
            hero.synergies.append("Team amplification")

        hero.power_level = target_power
        return hero


# ---------------- TEST ---------------- #

def test_balancer():
    balancer = PowerBalancer()

    heroes = [
        Hero(
            name="Chronos",
            abilities=["Time manipulation", "Temporal loops", "Age acceleration"],
            power_type="reality",
            power_level=0,
            weaknesses=[],
            synergies=[],
        ),
        Hero(
            name="Mindweaver",
            abilities=["Telepathy", "Illusion creation", "Memory manipulation"],
            power_type="mental",
            power_level=0,
            weaknesses=[],
            synergies=[],
        ),
        Hero(
            name="Quantum",
            abilities=["Teleportation", "Probability manipulation", "Phase shifting"],
            power_type="reality",
            power_level=0,
            weaknesses=[],
            synergies=[],
        ),
    ]

    print("⚡ SUPERHERO POWER BALANCER ⚡")
    print("=" * 70)

    for hero in heroes:
        analysis = balancer.analyze_hero_zero_shot(hero)
        print(f"\n🦸 Hero: {hero.name}")
        print(f"Power Level: {analysis['power_level']:.1f}/10")
        print(f"Power Type: {balancer.classify_power_few_shot(hero.abilities)}")
        print("-" * 70)

    print("\n🎯 BALANCE ANALYSIS:")
    print("=" * 70)

    report = balancer.detect_imbalance_combined(heroes[0], heroes)
    print(f"Hero: {report.hero.name}")
    print(f"Power Rating: {report.power_rating:.1f}/10")

    for issue in report.balance_issues:
        print("⚠️", issue)
    for change in report.suggested_changes:
        print("✓", change)


if __name__ == "__main__":
    if not os.environ.get("OPENAI_API_KEY"):
        print("⚠️ Please set OPENAI_API_KEY")
    else:
        test_balancer()
