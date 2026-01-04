"""
Assignment 9: Mystery Dinner Party Solver
All Prompting Techniques Combined
"""

import os
import json
from typing import Dict, List
from dataclasses import dataclass
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# ---------------- DATA CLASSES ---------------- #

@dataclass
class Suspect:
    name: str
    background: str
    alibi: str
    motive: str
    opportunity: bool
    suspicious_behavior: List[str]


@dataclass
class Clue:
    description: str
    location: str
    time_found: str
    related_suspects: List[str]
    significance: str


@dataclass
class MysteryCase:
    victim: str
    crime_scene: str
    time_of_death: str
    suspects: List[Suspect]
    clues: List[Clue]
    witness_statements: List[str]


@dataclass
class Solution:
    murderer: str
    motive: str
    method: str
    reasoning_chain: List[str]
    evidence_links: Dict[str, str]
    confidence: float
    alternative_theories: List[str]


# ---------------- DETECTIVE ---------------- #

class MysteryDetective:
    """
    AI detective using zero-shot, few-shot, and CoT prompting.
    """

    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.llm = ChatOpenAI(model=model_name, temperature=0.3)
        self._setup_chains()

    # ---------------- SETUP CHAINS ---------------- #

    def _setup_chains(self):

        # ZERO-SHOT: Psychological profiling
        self.profile_chain = (
            PromptTemplate.from_template(
                """
Psychologically analyze this murder suspect.

Tasks:
- Estimate deception likelihood (0–1)
- Estimate motive strength (0–1)
- Provide brief psychological profile

Suspect:
{suspect}

Respond ONLY in JSON:
{{
  "deception": 0.0,
  "motive_strength": 0.0,
  "profile": "..."
}}
"""
            )
            | self.llm
            | StrOutputParser()
        )

        # FEW-SHOT: Clue pattern analysis
        examples = [
            {
                "clue": "Lipstick on wine glass",
                "analysis": "Suggests close contact, possibly romantic or intimate",
                "importance": "high",
            },
            {
                "clue": "Broken watch at crime scene",
                "analysis": "Time discrepancy may indicate struggle",
                "importance": "medium",
            },
        ]

        self.clue_chain = (
            FewShotPromptTemplate(
                examples=examples,
                example_prompt=PromptTemplate.from_template(
                    "Clue: {clue}\nAnalysis: {analysis}\nImportance: {importance}\n"
                ),
                prefix="Analyze the following clue based on known patterns.\n\n",
                suffix="Clue: {clue}\nAnalysis:",
                input_variables=["clue"],
            )
            | self.llm
            | StrOutputParser()
        )

        # CoT: Timeline reconstruction
        self.timeline_chain = (
            PromptTemplate.from_template(
                """
Reconstruct the murder timeline step by step.

Alibis:
{alibis}

Witness Statements:
{witnesses}

Time of Death: {tod}

Determine which alibis are suspicious.

Respond ONLY in JSON:
{{
  "results": {{
    "Suspect Name": true/false
  }}
}}

Let's think step by step.
"""
            )
            | self.llm
            | StrOutputParser()
        )

        # FINAL SOLVER (combined)
        self.solve_chain = (
            PromptTemplate.from_template(
                """
Solve the murder mystery using all available information.

Victim: {victim}
Scene: {scene}
Time of Death: {tod}

Suspects:
{suspects}

Clues:
{clues}

Alibi Verification:
{alibis}

Provide final deduction.

Respond ONLY in JSON:
{{
  "murderer": "...",
  "motive": "...",
  "method": "...",
  "reasoning": ["step 1", "step 2"],
  "confidence": 0.0
}}
"""
            )
            | self.llm
            | StrOutputParser()
        )

    # ---------------- ZERO-SHOT PROFILING ---------------- #

    def profile_suspect(self, suspect: Suspect) -> Dict[str, any]:
        text = f"""
Name: {suspect.name}
Background: {suspect.background}
Alibi: {suspect.alibi}
Motive: {suspect.motive}
Behavior: {', '.join(suspect.suspicious_behavior)}
"""

        response = self.profile_chain.invoke({"suspect": text})
        cleaned = response.replace("```json", "").replace("```", "").strip()

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            data = {}

        return {
            "deception_likelihood": data.get("deception", 0.5),
            "motive_strength": data.get("motive_strength", 0.5),
            "psychological_profile": data.get("profile", ""),
        }

    # ---------------- FEW-SHOT CLUES ---------------- #

    def analyze_clues(self, clues: List[Clue]) -> List[Dict[str, any]]:
        results = []
        for clue in clues:
            analysis = self.clue_chain.invoke({"clue": clue.description})
            results.append(
                {
                    "clue": clue.description,
                    "analysis": analysis.strip(),
                    "suspects": clue.related_suspects,
                }
            )
        return results

    # ---------------- CoT ALIBI CHECK ---------------- #

    def verify_alibis(self, case: MysteryCase) -> Dict[str, bool]:
        alibis = {s.name: s.alibi for s in case.suspects}

        response = self.timeline_chain.invoke(
            {
                "alibis": json.dumps(alibis, indent=2),
                "witnesses": case.witness_statements,
                "tod": case.time_of_death,
            }
        )

        cleaned = response.replace("```json", "").replace("```", "").strip()

        try:
            return json.loads(cleaned)["results"]
        except Exception:
            return {s.name: False for s in case.suspects}

    # ---------------- FINAL SOLUTION ---------------- #

    def solve_mystery(self, case: MysteryCase) -> Solution:
        profiles = {
            s.name: self.profile_suspect(s) for s in case.suspects
        }
        alibi_check = self.verify_alibis(case)
        clues = self.analyze_clues(case.clues)

        response = self.solve_chain.invoke(
            {
                "victim": case.victim,
                "scene": case.crime_scene,
                "tod": case.time_of_death,
                "suspects": json.dumps(profiles, indent=2),
                "clues": json.dumps(clues, indent=2),
                "alibis": json.dumps(alibi_check, indent=2),
            }
        )

        cleaned = response.replace("```json", "").replace("```", "").strip()

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            data = {}

        return Solution(
            murderer=data.get("murderer", "Unknown"),
            motive=data.get("motive", ""),
            method=data.get("method", ""),
            reasoning_chain=data.get("reasoning", []),
            evidence_links={},
            confidence=data.get("confidence", 0.5),
            alternative_theories=[],
        )


# ---------------- TEST ---------------- #

def test_detective():
    detective = MysteryDetective()

    test_case = MysteryCase(
        victim="Lord Wellington",
        crime_scene="Library",
        time_of_death="10:30 PM",
        suspects=[
            Suspect(
                name="Lady Scarlett",
                background="Victim's wife, inherits estate",
                alibi="In the garden with guests",
                motive="Inheritance and secret affair",
                opportunity=True,
                suspicious_behavior=["Nervous", "Changed story twice"],
            ),
            Suspect(
                name="Professor Plum",
                background="Business partner, recent disputes",
                alibi="In study reviewing documents",
                motive="Business betrayal",
                opportunity=True,
                suspicious_behavior=["Destroyed papers after murder"],
            ),
            Suspect(
                name="Colonel Mustard",
                background="Old friend, owes money",
                alibi="Playing billiards with butler",
                motive="Gambling debts",
                opportunity=False,
                suspicious_behavior=["Attempted to leave early"],
            ),
        ],
        clues=[
            Clue(
                description="Poison bottle hidden in bookshelf",
                location="Library",
                time_found="11:00 PM",
                related_suspects=["Lady Scarlett", "Professor Plum"],
                significance="Murder weapon",
            ),
            Clue(
                description="Love letter from unknown person",
                location="Victim's pocket",
                time_found="10:45 PM",
                related_suspects=["Lady Scarlett"],
                significance="Possible motive",
            ),
        ],
        witness_statements=[
            "Butler saw Professor Plum near library at 10:15 PM",
            "Maid heard argument from library at 10:20 PM",
            "Guest saw Lady Scarlett in garden until 10:25 PM",
        ],
    )

    print("🕵️ MYSTERY DINNER PARTY SOLVER 🕵️")
    print("=" * 70)

    solution = detective.solve_mystery(test_case)
    print(f"Murderer: {solution.murderer}")
    print(f"Motive: {solution.motive}")
    print(f"Method: {solution.method}")
    print(f"Confidence: {solution.confidence:.0%}")


if __name__ == "__main__":
    if not os.environ.get("OPENAI_API_KEY"):
        print("⚠️ Please set OPENAI_API_KEY")
    else:
        test_detective()
