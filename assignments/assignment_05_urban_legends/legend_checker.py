"""
Assignment 5: Urban Legend Fact Checker
Zero-shot + Few-shot Prompting
"""

import os
import json
from typing import Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# ---------------- ENUMS ---------------- #

class MythCategory(Enum):
    SUPERNATURAL = "supernatural"
    CONSPIRACY = "conspiracy"
    MEDICAL = "medical_health"
    TECHNOLOGY = "technology"
    HISTORICAL = "historical"
    SOCIAL = "social_phenomena"
    CREATURE = "cryptid_creature"


class LogicalFallacy(Enum):
    FALSE_CAUSE = "false_cause"
    APPEAL_TO_AUTHORITY = "appeal_to_authority"
    HASTY_GENERALIZATION = "hasty_generalization"
    CIRCULAR_REASONING = "circular_reasoning"


# ---------------- DATA CLASSES ---------------- #

@dataclass
class Claim:
    text: str
    testable: bool
    evidence_required: str
    confidence: float


@dataclass
class MythAnalysis:
    original_text: str
    category: str
    claims: List[Claim]
    logical_fallacies: List[str]
    truth_rating: float
    believability_score: float
    debunking_explanation: str
    similar_myths: List[str]
    origin_theory: str


# ---------------- MAIN CLASS ---------------- #

class UrbanLegendChecker:

    def __init__(self, model_name="gpt-4o-mini", temperature=0.3):
        self.llm = ChatOpenAI(model=model_name, temperature=temperature)
        self._setup_chains()

    # ---------------- SETUP CHAINS ---------------- #

    def _setup_chains(self):

        # ZERO-SHOT: Claim extraction
        claim_template = PromptTemplate.from_template(
            """
Extract factual claims from the urban legend.

Rules:
- A claim must assert something that could be tested or verified
- For each claim provide:
  - text
  - whether it is testable
  - evidence required
  - confidence (0–1)

Respond ONLY with valid JSON.

JSON format:
{{
  "claims": [
    {{
      "text": "",
      "testable": true,
      "evidence_required": "",
      "confidence": 0.5
    }}
  ]
}}

Legend:
{legend_text}
"""
        )

        self.claim_extractor = claim_template | self.llm | StrOutputParser()

        # FEW-SHOT: Myth classification
        examples = [
            {
                "legend": "Alligators live in NYC sewers after being flushed as pets.",
                "category": "cryptid_creature",
                "reasoning": "Hidden creature living secretly in urban environment",
            },
            {
                "legend": "5G towers control human thoughts.",
                "category": "conspiracy",
                "reasoning": "Claims secret government control using technology",
            },
            {
                "legend": "Pop Rocks and soda can kill you.",
                "category": "medical_health",
                "reasoning": "False health danger claim",
            },
            {
                "legend": "A ghost hitchhiker vanishes from cars.",
                "category": "supernatural",
                "reasoning": "Involves ghosts and unexplained disappearance",
            },
        ]

        example_prompt = PromptTemplate.from_template(
            "Legend: {legend}\nCategory: {category}\nReasoning: {reasoning}\n"
        )

        self.myth_classifier = FewShotPromptTemplate(
            examples=examples,
            example_prompt=example_prompt,
            prefix="Classify the following urban legend.\n\n",
            suffix="Legend: {legend_text}\nCategory:",
            input_variables=["legend_text"],
        ) | self.llm | StrOutputParser()

        # ZERO-SHOT: Fallacy detection
        fallacy_template = PromptTemplate.from_template(
            """
Identify logical fallacies in this myth.

Common fallacies:
- False cause
- Appeal to authority
- Hasty generalization
- Circular reasoning

Respond with a JSON list.

JSON format:
["fallacy_name", "fallacy_name"]

Myth:
{legend_text}
"""
        )

        self.fallacy_detector = fallacy_template | self.llm | StrOutputParser()

        # ZERO-SHOT: Debunking
        debunk_template = PromptTemplate.from_template(
            """
Debunk this urban legend clearly and respectfully.

Guidelines:
- Use scientific reasoning
- Explain why people believe it
- Use simple language

Legend:
{legend_text}

Claims:
{claims}

Debunking:
"""
        )

        self.debunker = debunk_template | self.llm | StrOutputParser()

    # ---------------- TASK METHODS ---------------- #

    def extract_claims_zero_shot(self, legend_text: str) -> List[Claim]:
        response = self.claim_extractor.invoke({"legend_text": legend_text})
        data = json.loads(response)

        claims = []
        for c in data.get("claims", []):
            claims.append(
                Claim(
                    text=c["text"],
                    testable=c["testable"],
                    evidence_required=c["evidence_required"],
                    confidence=c["confidence"],
                )
            )
        return claims

    def classify_myth_few_shot(self, legend_text: str) -> Tuple[str, str]:
        category = self.myth_classifier.invoke({"legend_text": legend_text}).strip()
        return category, "Matched against known myth patterns"

    def detect_fallacies_combined(self, legend_text: str, claims: List[Claim]) -> List[str]:
        try:
            return json.loads(self.fallacy_detector.invoke({"legend_text": legend_text}))
        except json.JSONDecodeError:
            return []

    def calculate_believability(self, legend_text: str, claims: List[Claim], fallacies: List[str]) -> float:
        score = 0.6
        if fallacies:
            score -= 0.2
        if len(claims) > 2:
            score += 0.1
        return max(0.0, min(score, 1.0))

    def find_similar_myths(self, legend_text: str, category: str) -> List[str]:
        examples = {
            "supernatural": ["Vanishing hitchhiker", "Resurrection Mary"],
            "conspiracy": ["Chemtrails", "Mind control experiments"],
            "medical_health": ["Vaccines cause autism", "Microwaves cause cancer"],
            "cryptid_creature": ["Sewer alligators", "Bigfoot sightings"],
        }
        return examples.get(category, [])

    def analyze_legend(self, legend_text: str) -> MythAnalysis:
        claims = self.extract_claims_zero_shot(legend_text)
        category, _ = self.classify_myth_few_shot(legend_text)
        fallacies = self.detect_fallacies_combined(legend_text, claims)
        believability = self.calculate_believability(legend_text, claims, fallacies)

        debunking = self.debunker.invoke({
            "legend_text": legend_text,
            "claims": ", ".join(c.text for c in claims),
        })

        truth_rating = 0.1 if fallacies else 0.4

        return MythAnalysis(
            original_text=legend_text,
            category=category,
            claims=claims,
            logical_fallacies=fallacies,
            truth_rating=truth_rating,
            believability_score=believability,
            debunking_explanation=debunking,
            similar_myths=self.find_similar_myths(legend_text, category),
            origin_theory="Likely spread through oral storytelling and viral sharing",
        )

    def adaptive_analysis(self, legend_text: str) -> Dict[str, any]:
        return {
            "analysis": asdict(self.analyze_legend(legend_text)),
            "method_choices": {
                "claim_extraction": "zero-shot",
                "classification": "few-shot",
                "fallacy_detection": "zero-shot",
            },
            "reasoning": "Used few-shot for pattern recognition and zero-shot for open analysis",
        }


# ---------------- TEST ---------------- #

def test_legend_checker():
    checker = UrbanLegendChecker()

    legends = [
        "5G towers control human thoughts and emotions.",
        "A ghost hitchhiker disappears from the back seat.",
    ]

    print("🕵️ URBAN LEGEND FACT CHECKER 🕵️")
    print("=" * 70)

    for text in legends:
        analysis = checker.analyze_legend(text)
        print("\nLegend:", text)
        print("Category:", analysis.category)
        print("Truth Rating:", f"{analysis.truth_rating:.0%}")
        print("Believability:", f"{analysis.believability_score:.0%}")
        print("Debunking:", analysis.debunking_explanation[:120], "...")
        print("-" * 70)


if __name__ == "__main__":
    if not os.environ.get("OPENAI_API_KEY"):
        print("⚠️ Please set OPENAI_API_KEY environment variable")
    else:
        test_legend_checker()
