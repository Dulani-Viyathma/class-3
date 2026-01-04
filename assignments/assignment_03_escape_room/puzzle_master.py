import os
import json
import random
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum

# LangChain Imports
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# --- Data Structures ---

class PuzzleType(Enum):
    RIDDLE = "riddle"
    CIPHER = "cipher"
    LOGIC = "logic"
    PATTERN = "pattern"
    WORDPLAY = "wordplay"

class DifficultyLevel(Enum):
    BEGINNER = 1
    EASY = 2
    MEDIUM = 3
    HARD = 4
    EXPERT = 5

@dataclass
class Puzzle:
    puzzle_text: str
    solution: str
    puzzle_type: str
    difficulty: int
    hints: List[str]
    explanation: str
    time_estimate: int

@dataclass
class PuzzleSequence:
    theme: str
    puzzles: List[Puzzle]
    final_solution: str
    narrative: str

# --- The Puzzle Master Class ---

class PuzzleMaster:
    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.7):
        self.llm = ChatOpenAI(model=model_name, temperature=temperature)
        self.puzzle_examples = self._load_puzzle_examples()
        self._setup_chains()

    def _load_puzzle_examples(self) -> Dict[str, List[dict]]:
        """TODO #1: Examples for few-shot learning."""
        return {
            "riddle": [
                {
                    "puzzle": "I have cities, but no houses. I have mountains, but no trees. I have water, but no fish. What am I?",
                    "solution": "A map",
                    "difficulty": "2",
                    "explanation": "A map depicts geography without being the physical objects themselves."
                },
                {
                    "puzzle": "What has to be broken before you can use it?",
                    "solution": "An egg",
                    "difficulty": "1",
                    "explanation": "To cook or eat an egg, the shell must be cracked."
                }
            ],
            "cipher": [
                {
                    "puzzle": "Decode: 1-20-20-1-3-11",
                    "solution": "ATTACK",
                    "difficulty": "3",
                    "explanation": "A simple A=1, B=2 alpha-numeric substitution."
                }
            ],
            "logic": [
                {
                    "puzzle": "A man is looking at a photograph. His friend asks, 'Who is it?' The man replies, 'Brothers and sisters, I have none. But that man's father is my father's son.' Who is in the photograph?",
                    "solution": "His son",
                    "difficulty": "4",
                    "explanation": "My father's son (with no siblings) is ME. So, that man's father is ME."
                }
            ],
            "pattern": [
                {
                    "puzzle": "1, 1, 2, 3, 5, 8, ?",
                    "solution": "13",
                    "difficulty": "2",
                    "explanation": "Fibonacci sequence: each number is the sum of the previous two."
                }
            ]
        }

    def _setup_chains(self):
        """TODO #2: Setup logic chains."""
        self.parser = JsonOutputParser()
        
        # Base formatter for examples
        self.example_prompt = PromptTemplate(
            input_variables=["puzzle", "solution", "difficulty", "explanation"],
            template="Puzzle: {puzzle}\nSolution: {solution}\nDifficulty: {difficulty}\nExplanation: {explanation}"
        )

        # Validation Prompt
        val_template = """As an Escape Room Auditor, evaluate this puzzle for quality and solvability.
        Puzzle: {puzzle_text}
        Solution: {solution}
        
        Respond ONLY in JSON format:
        {{
            "is_solvable": bool,
            "has_unique_solution": bool,
            "difficulty_appropriate": bool,
            "issues": [string],
            "suggestions": [string]
        }}"""
        self.validation_chain = PromptTemplate.from_template(val_template) | self.llm | self.parser

        # Hint Prompt
        hint_template = """Create {num_hints} progressive hints for this puzzle.
        Puzzle: {puzzle_text}
        Solution: {solution}
        
        Format as a JSON list of strings, from subtle to very obvious."""
        self.hint_chain = PromptTemplate.from_template(hint_template) | self.llm | self.parser

    def generate_puzzle(self, puzzle_type: PuzzleType, difficulty: DifficultyLevel, theme: str = "general") -> Puzzle:
        """TODO #3: Generate puzzle using Few-Shot Prompting."""
        examples = self.puzzle_examples.get(puzzle_type.value, [])
        
        few_shot_template = FewShotPromptTemplate(
            examples=examples,
            example_prompt=self.example_prompt,
            prefix=f"You are a master escape room designer. Create a {puzzle_type.value} puzzle with a '{theme}' theme.",
            suffix="Now create a new puzzle for difficulty level {difficulty_val}. Return JSON with keys: puzzle_text, solution, explanation, time_estimate.",
            input_variables=["difficulty_val"]
        )

        chain = few_shot_template | self.llm | self.parser
        raw = chain.invoke({"difficulty_val": difficulty.value})

        # Create the initial puzzle object
        puzzle = Puzzle(
            puzzle_text=raw["puzzle_text"],
            solution=raw["solution"],
            puzzle_type=puzzle_type.value,
            difficulty=difficulty.value,
            hints=[],
            explanation=raw["explanation"],
            time_estimate=raw.get("time_estimate", 5)
        )
        
        # Auto-generate hints for the puzzle
        puzzle.hints = self.generate_hints(puzzle)
        return puzzle

    def validate_puzzle(self, puzzle: Puzzle) -> Dict[str, Any]:
        """TODO #4: Validate puzzle solvability."""
        return self.validation_chain.invoke({
            "puzzle_text": puzzle.puzzle_text,
            "solution": puzzle.solution,
            "difficulty": puzzle.difficulty
        })

    def generate_hints(self, puzzle: Puzzle, num_hints: int = 3) -> List[str]:
        """TODO #5: Generate hint progression."""
        return self.hint_chain.invoke({
            "puzzle_text": puzzle.puzzle_text,
            "solution": puzzle.solution,
            "num_hints": num_hints
        })

    def create_puzzle_sequence(self, theme: str, num_puzzles: int = 3, difficulty_curve: str = "increasing") -> PuzzleSequence:
        """TODO #6: Build an interconnected sequence."""
        puzzles = []
        puzzle_types = list(PuzzleType)
        
        for i in range(num_puzzles):
            # Calculate difficulty based on index
            diff_score = min(i + 2, 5) if difficulty_curve == "increasing" else 3
            diff = DifficultyLevel(diff_score)
            p_type = random.choice(puzzle_types)
            puzzles.append(self.generate_puzzle(p_type, diff, theme))

        return PuzzleSequence(
            theme=theme,
            puzzles=puzzles,
            final_solution=puzzles[-1].solution,
            narrative=f"You find yourself trapped in a {theme}. To escape, you must solve a series of trials."
        )

# --- Test Script ---

def test_puzzle_master():
    if not os.environ.get("OPENAI_API_KEY"):
        print("⚠️ Please set OPENAI_API_KEY")
        return

    master = PuzzleMaster()
    print("🔐 ESCAPE ROOM PUZZLE MASTER 🔐\n" + "="*40)

    # Test Single Generation
    p = master.generate_puzzle(PuzzleType.RIDDLE, DifficultyLevel.EASY, "Ancient Egypt")
    print(f"Theme: Ancient Egypt | Type: Riddle")
    print(f"Puzzle: {p.puzzle_text}")
    print(f"Solution: {p.solution}")
    print(f"Hints: {p.hints}")

    # Test Validation
    val = master.validate_puzzle(p)
    print(f"Valid: {val['is_solvable']}")

if __name__ == "__main__":
    test_puzzle_master()