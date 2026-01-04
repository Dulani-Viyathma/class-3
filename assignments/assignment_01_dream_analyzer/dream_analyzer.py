import os
import json
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, asdict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# ---------------- DATA CLASSES ---------------- #

@dataclass
class DreamSymbol:
    symbol: str
    meaning: str
    frequency: int = 1
    significance: float = 0.5

@dataclass
class DreamAnalysis:
    symbols: List[DreamSymbol]
    emotions: List[str]
    themes: List[str]
    lucidity_score: float
    psychological_insights: str
    recurring_patterns: List[str]
    dream_type: str

# ---------------- ANALYZER ---------------- #

class DreamAnalyzer:
    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.3):
        self.llm = ChatOpenAI(model=model_name, temperature=temperature)
        self.parser = JsonOutputParser()
        self._setup_chains()

    def _setup_chains(self):
        # Using {{ }} to escape JSON brackets for LangChain
        symbol_template = PromptTemplate.from_template(
            """Extract symbols from this dream. Return strictly JSON.
            {{
                "symbols": [
                    {{"symbol": "name", "meaning": "interpretation", "frequency": 1, "significance": 0.5}}
                ]
            }}
            Dream: {dream_text}"""
        )

        emotion_template = PromptTemplate.from_template(
            """Identify emotions felt in this dream. Return strictly JSON.
            {{
                "emotions": ["joy", "confusion"],
                "intensity": 5.0
            }}
            Dream: {dream_text}"""
        )

        insight_template = PromptTemplate.from_template(
            """Analyze themes, patterns, lucidity (0-10), type, and insights. Return strictly JSON.
            {{
                "themes": [],
                "recurring_patterns": [],
                "lucidity_score": 5.0,
                "dream_type": "normal",
                "psychological_insights": ""
            }}
            Dream: {dream_text}
            Symbols: {symbols}
            Emotions: {emotions}"""
        )

        self.symbol_chain = symbol_template | self.llm | self.parser
        self.emotion_chain = emotion_template | self.llm | self.parser
        self.insight_chain = insight_template | self.llm | self.parser

    def extract_symbols(self, dream_text: str) -> List[DreamSymbol]:
        data = self.symbol_chain.invoke({"dream_text": dream_text})
        symbols = []
        for s in data.get("symbols", []):
            # Ensure frequency/significance are the right types
            symbols.append(DreamSymbol(
                symbol=str(s.get("symbol", "unknown")),
                meaning=str(s.get("meaning", "no interpretation")),
                frequency=int(s.get("frequency", 1)),
                significance=float(s.get("significance", 0.5))
            ))
        return symbols

    def analyze_emotions(self, dream_text: str) -> Tuple[List[str], float]:
        data = self.emotion_chain.invoke({"dream_text": dream_text})
        raw_emotions = data.get("emotions", [])
        
        # Robustly handle if LLM returns a list of dicts instead of list of strings
        emotions = []
        for e in raw_emotions:
            if isinstance(e, dict):
                emotions.append(next(iter(e.values()))) # Get first value in dict
            else:
                emotions.append(str(e))
                
        intensity = float(data.get("intensity", 5.0))
        return emotions, intensity

    def generate_insights(self, dream_text: str, symbols: List[DreamSymbol], emotions: List[str]) -> Dict:
        sym_str = ", ".join([s.symbol for s in symbols])
        emo_str = ", ".join(emotions) # Now safe because analyze_emotions sanizited it
        
        return self.insight_chain.invoke({
            "dream_text": dream_text,
            "symbols": sym_str,
            "emotions": emo_str
        })

    def analyze_dream(self, dream_text: str) -> DreamAnalysis:
        # Step-by-step pipeline
        symbols = self.extract_symbols(dream_text)
        emotions, _ = self.analyze_emotions(dream_text)
        insight_data = self.generate_insights(dream_text, symbols, emotions)

        return DreamAnalysis(
            symbols=symbols,
            emotions=emotions,
            themes=insight_data.get("themes", []),
            lucidity_score=float(insight_data.get("lucidity_score", 0.0)),
            psychological_insights=insight_data.get("psychological_insights", ""),
            recurring_patterns=insight_data.get("recurring_patterns", []),
            dream_type=insight_data.get("dream_type", "normal")
        )

# ---------------- TEST RUNNER ---------------- #

def test_dream_analyzer():
    analyzer = DreamAnalyzer()
    
    # Using the dream from your example logs
    dream = "I realized I was dreaming and could fly above my old school while feeling anxious but excited."
    
    try:
        analysis_obj = analyzer.analyze_dream(dream)
        # Convert dataclass to dict and print as pretty JSON
        print(json.dumps(asdict(analysis_obj), indent=2))
    except Exception as e:
        print(f"❌ Error during analysis: {e}")

if __name__ == "__main__":
    if not os.environ.get("OPENAI_API_KEY"):
        print("⚠️ Please set OPENAI_API_KEY environment variable")
    else:
        test_dream_analyzer()