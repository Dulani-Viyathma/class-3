"""
Assignment 10: AI Dungeon Master
The Ultimate Challenge – All Prompting Techniques Combined
"""

import os
import json
import random
from typing import Dict, List, Any
from dataclasses import dataclass, asdict
from enum import Enum
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# ---------------- ENUMS ---------------- #

class QuestType(Enum):
    RESCUE = "rescue"
    FETCH = "fetch"
    INVESTIGATE = "investigate"
    COMBAT = "combat"
    DIPLOMACY = "diplomacy"
    EXPLORATION = "exploration"


# ---------------- DATA CLASSES ---------------- #

@dataclass
class Character:
    name: str
    class_type: str
    level: int
    hit_points: int
    abilities: List[str]
    inventory: List[str]
    personality: str


@dataclass
class NPC:
    name: str
    role: str
    personality: str
    motivation: str
    dialogue_style: str
    secrets: List[str]


@dataclass
class Quest:
    title: str
    description: str
    objectives: List[str]
    rewards: List[str]
    difficulty: int
    quest_type: str


@dataclass
class CombatState:
    participants: List[Character]
    turn_order: List[str]
    environment: str
    special_conditions: List[str]


@dataclass
class WorldState:
    location: str
    time_of_day: str
    weather: str
    active_quests: List[Quest]
    npcs_present: List[NPC]
    recent_events: List[str]
    player_reputation: Dict[str, int]


# ---------------- DUNGEON MASTER ---------------- #

class DungeonMasterAI:
    """
    AI Dungeon Master using all prompting techniques.
    """

    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.llm = ChatOpenAI(model=model_name, temperature=0.7)
        self._setup_chains()

        self.world_state = WorldState(
            location="Tavern",
            time_of_day="Evening",
            weather="Clear",
            active_quests=[],
            npcs_present=[],
            recent_events=[],
            player_reputation={},
        )

    # ---------------- SETUP CHAINS ---------------- #

    def _setup_chains(self):

        # ZERO-SHOT: Story & quest generation
        self.story_chain = (
            PromptTemplate.from_template(
                """
Create a Dungeons & Dragons quest.

Requirements:
- Clear title
- Immersive description
- 2–4 objectives
- Appropriate rewards for party level
- Balanced difficulty

Quest Type: {quest_type}
Party Level: {level}

Respond ONLY in JSON:
{{
  "title": "...",
  "description": "...",
  "objectives": ["..."],
  "rewards": ["..."]
}}
"""
            )
            | self.llm
            | StrOutputParser()
        )

        # FEW-SHOT: NPC roleplay
        npc_examples = [
            {
                "npc": "Gruff Innkeeper",
                "response": "Ale's cheap, trouble's expensive. What do you want?",
            },
            {
                "npc": "Mysterious Noble",
                "response": "Power always comes at a price… are you prepared to pay it?",
            },
        ]

        self.npc_chain = (
            FewShotPromptTemplate(
                examples=npc_examples,
                example_prompt=PromptTemplate.from_template(
                    "NPC Type: {npc}\nDialogue: {response}\n"
                ),
                prefix="Roleplay the NPC below in character.\n\n",
                suffix="""
NPC Name: {name}
Personality: {personality}
Motivation: {motivation}
Player Says: {player_input}

NPC Response:
""",
                input_variables=["name", "personality", "motivation", "player_input"],
            )
            | self.llm
            | StrOutputParser()
        )

        # CoT: Combat resolution
        self.combat_chain = (
            PromptTemplate.from_template(
                """
Resolve this combat action step by step.

Action: {action}
Attacker Level: {level}
Target HP: {hp}
Environment: {environment}

Step 1: Roll to hit (assume d20)
Step 2: Determine hit or miss
Step 3: Calculate damage if hit

Respond ONLY in JSON:
{{
  "hit": true/false,
  "damage": 0,
  "description": "..."
}}
"""
            )
            | self.llm
            | StrOutputParser()
        )

        # COMBINED: World update
        self.world_chain = (
            PromptTemplate.from_template(
                """
Update the world state logically and creatively.

Current State:
{state}

Player Actions:
{actions}

Time Passed: {time}

Respond ONLY in JSON:
{{
  "location": "...",
  "time_of_day": "...",
  "recent_events": ["..."]
}}
"""
            )
            | self.llm
            | StrOutputParser()
        )

    # ---------------- QUEST ---------------- #

    def generate_quest(self, quest_type: QuestType, party_level: int) -> Quest:
        response = self.story_chain.invoke(
            {"quest_type": quest_type.value, "level": party_level}
        )

        cleaned = response.replace("```json", "").replace("```", "").strip()
        data = json.loads(cleaned)

        return Quest(
            title=data["title"],
            description=data["description"],
            objectives=data["objectives"],
            rewards=data["rewards"],
            difficulty=party_level,
            quest_type=quest_type.value,
        )

    # ---------------- NPC ---------------- #

    def roleplay_npc(self, npc: NPC, player_input: str, context: Dict[str, Any]) -> str:
        return self.npc_chain.invoke(
            {
                "name": npc.name,
                "personality": npc.personality,
                "motivation": npc.motivation,
                "player_input": player_input,
            }
        )

    # ---------------- COMBAT ---------------- #

    def resolve_combat(
        self,
        action: str,
        attacker: Character,
        target: Character,
        combat_state: CombatState,
    ) -> Dict[str, Any]:

        response = self.combat_chain.invoke(
            {
                "action": action,
                "level": attacker.level,
                "hp": target.hit_points,
                "environment": combat_state.environment,
            }
        )

        cleaned = response.replace("```json", "").replace("```", "").strip()
        return json.loads(cleaned)

    # ---------------- NARRATION ---------------- #

    def narrate_scene(
        self, action: str, world_state: WorldState, characters: List[Character]
    ) -> str:
        return (
            f"As {action.lower()}, the air shifts with tension. "
            f"The {world_state.location} feels alive with unseen danger."
        )

    # ---------------- WORLD UPDATE ---------------- #

    def update_world(self, actions: List[str], time_passed: str) -> WorldState:
        response = self.world_chain.invoke(
            {
                "state": json.dumps(asdict(self.world_state)),
                "actions": actions,
                "time": time_passed,
            }
        )

        cleaned = response.replace("```json", "").replace("```", "").strip()
        data = json.loads(cleaned)

        self.world_state.location = data["location"]
        self.world_state.time_of_day = data["time_of_day"]
        self.world_state.recent_events.extend(data["recent_events"])

        return self.world_state

    # ---------------- SESSION ---------------- #

    def run_session(self, player_actions: List[str], party: List[Character]) -> Dict[str, Any]:
        narration = []
        for action in player_actions:
            narration.append(self.narrate_scene(action, self.world_state, party))

        self.update_world(player_actions, "1 hour")

        return {"narration": narration}


# ---------------- TEST ---------------- #

def test_dungeon_master():
    dm = DungeonMasterAI()

    party = [
        Character("Aldric", "Fighter", 3, 28, ["Second Wind"], ["Sword"], "Reckless"),
        Character("Lyra", "Wizard", 3, 18, ["Fireball"], ["Spellbook"], "Analytical"),
    ]

    npc = NPC(
        name="Gareth",
        role="Innkeeper",
        personality="Gruff but kind",
        motivation="Protect tavern",
        dialogue_style="Blunt",
        secrets=["Former adventurer"],
    )

    print("🎲 AI DUNGEON MASTER 🎲")
    print("=" * 70)

    quest = dm.generate_quest(QuestType.RESCUE, 3)
    print("Quest:", quest.title)

    print("\nNPC Interaction:")
    print("Gareth:", dm.roleplay_npc(npc, "Any work available?", {}))

    combat = dm.resolve_combat(
        "Aldric attacks the goblin",
        party[0],
        Character("Goblin", "Monster", 1, 7, [], [], "Cowardly"),
        CombatState(party, [], "Forest", []),
    )

    print("\nCombat Result:", combat)

    session = dm.run_session(
        ["Investigate cellar", "Open hidden door"], party
    )

    print("\nSession Highlights:")
    for line in session["narration"]:
        print("•", line)

    print("\n🏆 Adventure Continues...")


if __name__ == "__main__":
    if not os.environ.get("OPENAI_API_KEY"):
        print("⚠️ Please set OPENAI_API_KEY")
    else:
        test_dungeon_master()
