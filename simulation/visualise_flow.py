#!/usr/bin/env python3
"""
Cinquillo 2.0 LaTeX/TikZ match visualizer.

This version:
- runs exactly ONE chosen variant
- defaults to Baseline if no variant is passed
- accepts exactly 4 agents from the CLI
- generates LaTeX in the same fixed-column visual style as the manual template:
    * one fixed column per player
    * round blocks with separators
    * stacked same-player turns
    * horizontal, vertical, and wrap arrows
    * exact turncard macro structure

Examples
--------
python simulation/visualise_flow.py
python simulation/visualise_flow.py "combo rush"
python simulation/visualise_flow.py "intel war" --agents mcts-deep rl balanced aggressive
python simulation/visualise_flow.py --list-variants
python simulation/visualise_flow.py --list-agents
"""

import argparse
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Sequence, TextIO, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game.entities import (
    Card,
    GameState,
    GoodDiceEffect,
    BadDiceEffect,
    MatchEndMode,
    ScoringMode,
    Suit,
    VariantConfig,
)
from game.rules import Rules, PlayCard, RollDice, Pass
from agents.base_agents import (
    Agent,
    RandomAgent,
    create_aggressive_heuristic,
    create_balanced_heuristic,
    create_defensive_heuristic,
)
from agents.mcts_agent import (
    MCTSAgentDeep,
    MCTSAgentFast,
    MCTSAgentStandard,
    MCTSAgentSuperFast,
)
from agents.rl_agent import RLAgent


  
# Variant registry
  

@dataclass(frozen=True)
class VariantSpec:
    name: str
    good: str
    bad: str
    p_good: float
    penalty: int
    scoring: str
    end: str


VARIANT_SPECS: List[VariantSpec] = [
    VariantSpec("Baseline",         "Wild",         "Take 2",      0.50, 1, "WTA", "5R"),
    VariantSpec("Blitz",            "Wild",         "Take 2",      0.50, 2, "WTA", "1R"),
    VariantSpec("Card Exchange",    "Transfer",     "-2 pts",      0.50, 1, "DP",  "10 pts"),
    VariantSpec("Card Flood",       "Wild",         "Take 4",      0.50, 1, "WTA", "5R"),
    VariantSpec("Chaos Mode",       "Wild",         "-5 pts",      0.90, 3, "DP",  "20 pts"),
    VariantSpec("Combo Rush",       "Double Play",  "Forced Pass", 0.50, 2, "DP",  "5R"),
    VariantSpec("Double Edge",      "Double Play",  "Take 2",      0.50, 0, "WTA", "5R"),
    VariantSpec("Double Spy",       "Double Play",  "Skip",        0.60, 1, "WTA", "5R"),
    VariantSpec("Endurance",        "Wild",         "Take 2",      0.50, 1, "WTA", "15R"),
    VariantSpec("Fortune's Wheel",  "Wild",         "Forced Pass", 1.00, 1, "WTA", "5R"),
    VariantSpec("Gambler's Run",    "Double Play",  "-2 pts",      0.45, 2, "DP",  "20 pts"),
    VariantSpec("Ghost Hand",       "Transfer",     "Exposed",     0.40, 2, "DP",  "15 pts"),
    VariantSpec("Glass Cannon",     "Double Play",  "-5 pts",      0.30, 2, "DP",  "5R"),
    VariantSpec("Hand Swap",        "Transfer",     "Take 2",      0.55, 1, "WTA", "5R"),
    VariantSpec("Heavy Toll",       "Wild",         "-3 pts",      0.40, 3, "DP",  "20 pts"),
    VariantSpec("High Roller",      "Wild",         "-2 pts",      0.60, 2, "DP",  "15 pts"),
    VariantSpec("Intel War",        "Information",  "Exposed",     0.50, 1, "DP",  "15 pts"),
    VariantSpec("Lucky Draw",       "Wild",         "-1 pt",       0.80, 1, "WTA", "5R"),
    VariantSpec("Marathon",         "Wild",         "Take 2",      0.50, 1, "WTA", "10R"),
    VariantSpec("Mirror Match",     "Transfer",     "Skip",        0.50, 1, "WTA", "5R"),
    VariantSpec("Open Book",        "Information",  "Exposed",     0.50, 0, "WTA", "8R"),
    VariantSpec("Pass & Peek",      "Information",  "Take 2",      0.45, 2, "WTA", "5R"),
    VariantSpec("Point Race",       "Wild",         "-1 pt",       0.50, 1, "WTA", "20 pts"),
    VariantSpec("Power Play",       "Double Play",  "-3 pts",      0.70, 2, "DP",  "5R"),
    VariantSpec("Pure Strategy",    "Wild",         "Forced Pass", 0.00, 1, "WTA", "5R"),
    VariantSpec("Reveal Rush",      "Double Play",  "Exposed",     0.60, 1, "WTA", "5R"),
    VariantSpec("Risk & Reward",    "Wild",         "-4 pts",      0.50, 3, "DP",  "25 pts"),
    VariantSpec("Safe Harbour",     "Wild",         "Forced Pass", 0.50, 0, "WTA", "5R"),
    VariantSpec("Score Doubler",    "Wild",         "Take 2",      0.50, 1, "DP",  "15 pts"),
    VariantSpec("Scout's Edge",     "Information",  "Forced Pass", 0.70, 0, "WTA", "5R"),
    VariantSpec("Slow Burn",        "Wild",         "Take 2",      0.50, 0, "WTA", "20R"),
    VariantSpec("Sprint",           "Wild",         "Take 2",      0.50, 1, "WTA", "3R"),
    VariantSpec("Spy Game",         "Information",  "Skip",        0.65, 1, "WTA", "10 pts"),
]


def slugify(text: str) -> str:
    return (
        text.strip()
        .lower()
        .replace("&", "and")
        .replace("'", "")
        .replace(".", "")
        .replace("—", "-")
        .replace("–", "-")
        .replace("_", "-")
        .replace(" ", "-")
    )


VARIANT_REGISTRY: Dict[str, VariantSpec] = {
    slugify(spec.name): spec for spec in VARIANT_SPECS
}


def parse_good_effect(label: str) -> GoodDiceEffect:
    key = slugify(label)
    mapping = {
        "wild": GoodDiceEffect.WILD,
        "double-play": GoodDiceEffect.DOUBLE_PLAY,
        "transfer": GoodDiceEffect.GIVE_CARD,
        "information": GoodDiceEffect.INFO_REVEAL,
    }
    if key not in mapping:
        raise ValueError(f"Unknown good effect label: {label}")
    return mapping[key]


def parse_bad_effect(label: str) -> Tuple[BadDiceEffect, int]:
    raw = label.strip().lower()

    if raw.startswith("take"):
        count = int(raw.replace("take", "").strip().split()[0])
        return (BadDiceEffect.TAKE_CARDS, count)

    if raw.startswith("-") and "pt" in raw:
        number = raw.replace("pts", "").replace("pt", "").strip().lstrip("-")
        return (BadDiceEffect.NEGATIVE_POINTS, int(float(number)))

    if raw in {"forced pass", "skip"}:
        return (BadDiceEffect.FORCED_PASS, 1)

    if raw == "exposed":
        return (BadDiceEffect.REVEAL_HAND, 1)

    raise ValueError(f"Unknown bad effect label: {label}")


def parse_scoring(label: str) -> ScoringMode:
    key = label.strip().upper()
    if key == "WTA":
        return ScoringMode.WINNER_TAKES_ALL
    if key == "DP":
        return ScoringMode.DOUBLE_PENALTY
    raise ValueError(f"Unknown scoring label: {label}")


def parse_end(label: str) -> Tuple[MatchEndMode, int, int]:
    raw = label.strip().lower()
    if raw.endswith("r"):
        rounds = int(raw[:-1].strip())
        return (MatchEndMode.FIXED_ROUNDS, 10, rounds)
    if raw.endswith("pts"):
        points = int(float(raw[:-3].strip()))
        multiplier = max(1, points // 4)
        return (MatchEndMode.TARGET_SCORE, multiplier, 5)
    raise ValueError(f"Unknown end label: {label}")


def build_variant(spec: VariantSpec) -> VariantConfig:
    good_effect = parse_good_effect(spec.good)
    bad_effect, magnitude = parse_bad_effect(spec.bad)
    match_end_mode, target_score_multiplier, fixed_rounds_count = parse_end(spec.end)

    kwargs = {
        "dice_good_probability": spec.p_good,
        "dice_good_effect": good_effect,
        "dice_bad_effect": bad_effect,
        "scoring_mode": parse_scoring(spec.scoring),
        "points_per_card": 1,
        "voluntary_pass_penalty": spec.penalty,
        "match_end_mode": match_end_mode,
        "target_score_multiplier": target_score_multiplier,
        "fixed_rounds_count": fixed_rounds_count,
        "dice_bad_cards_count": 2,
        "dice_bad_penalty_points": 1,
    }

    if bad_effect == BadDiceEffect.TAKE_CARDS:
        kwargs["dice_bad_cards_count"] = magnitude
    elif bad_effect == BadDiceEffect.NEGATIVE_POINTS:
        kwargs["dice_bad_penalty_points"] = magnitude

    return VariantConfig(**kwargs)


  
# Agents
  

AGENT_BUILDERS = {
    "mcts-deep": lambda: MCTSAgentDeep(name="MCTS-Deep"),
    "mcts": lambda: MCTSAgentStandard(name="MCTS"),
    "mcts-fast": lambda: MCTSAgentFast(name="MCTS-Fast"),
    "mcts-superfast": lambda: MCTSAgentSuperFast(name="MCTS-SuperFast"),
    "rl": lambda: RLAgent(name="RL"),
    "balanced": create_balanced_heuristic,
    "aggressive": create_aggressive_heuristic,
    "defensive": create_defensive_heuristic,
    "random": lambda: RandomAgent(name="Random"),
    "random-safe": lambda: RandomAgent(name="Random-Safe", avoid_bad_moves=True),
}

DEFAULT_AGENT_KEYS = ["mcts", "rl", "balanced", "aggressive"]


def build_agents(agent_keys: Sequence[str]) -> List[Agent]:
    if len(agent_keys) != 4:
        raise ValueError("You must provide exactly 4 agents.")
    agents: List[Agent] = []
    for key in agent_keys:
        norm = slugify(key)
        if norm not in AGENT_BUILDERS:
            raise ValueError(
                f"Unknown agent '{key}'. Available: {', '.join(sorted(AGENT_BUILDERS.keys()))}"
            )
        agents.append(AGENT_BUILDERS[norm]())
    return agents


  
# Visualizer
  

class LaTeXGridVisualizer:
    """
    Generates one fixed-column LaTeX/TikZ visualization matching the hand-written style.
    """

    SUIT_TO_LATEX = {
        Suit.OROS: 1,
        Suit.COPAS: 2,
        Suit.ESPADAS: 3,
        Suit.BASTOS: 4,
    }

    SUIT_NAMES = {
        Suit.OROS: "Oros",
        Suit.COPAS: "Copas",
        Suit.ESPADAS: "Espadas",
        Suit.BASTOS: "Bastos",
    }

    RANK_TO_INDEX = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 10: 8, 11: 9, 12: 10}

    COL_X = [2.20, 9.60, 17.00, 24.40]
    TURN_W = 7.00
    INTRA_GAP = 0.25
    INTER_ROUND_GAP = 0.80

    def __init__(
        self,
        agents: List[Agent],
        variant: VariantConfig,
        variant_name: str,
        output_file: Optional[str] = None,
    ):
        self.agents = agents
        self.variant = variant
        self.variant_name = variant_name
        self.output_file = output_file
        self.file_handle: Optional[TextIO] = None
        self.turn_states: List[dict] = []

      
    # Public
      

    def play_and_visualize(self) -> None:
        state = Rules.initialize_game(len(self.agents), self.variant)

        turn_count = 0
        max_turns = 1000

        self.turn_states.append({
            "turn": 0,
            "state": state.copy(),
            "agent": None,
            "move": None,
            "legal_moves": [],
            "post_state": None,
        })

        while not Rules.is_terminal(state) and turn_count < max_turns:
            turn_count += 1
            current_player_idx = state.current_player
            agent = self.agents[current_player_idx]
            legal_moves = Rules.get_legal_moves(state)
            chosen_move = agent.choose_move(state, legal_moves)

            pre_state = state.copy()
            state = chosen_move.apply(state)

            self.turn_states.append({
                "turn": turn_count,
                "state": pre_state,
                "post_state": state.copy(),
                "agent": agent,
                "move": chosen_move,
                "legal_moves": legal_moves,
            })

        Rules.compute_round_scores(state)
        self.turn_states.append({
            "turn": turn_count + 1,
            "state": state,
            "agent": None,
            "move": None,
            "legal_moves": [],
            "post_state": None,
            "final": True,
        })

        self._generate_latex()

      
    # IO
      

    def _write(self, text: str = "") -> None:
        if self.file_handle:
            self.file_handle.write(text + "\n")
        else:
            print(text)

    def _generate_latex(self) -> None:
        if self.output_file:
            self.file_handle = open(self.output_file, "w", encoding="utf-8")

        try:
            self._write_preamble()
            self._write_document()
            self._write_postamble()
            if self.output_file:
                print(f"✓ LaTeX visualization saved to: {self.output_file}")
                print(f"  Compile with: pdflatex {self.output_file}")
        finally:
            if self.file_handle:
                self.file_handle.close()

      
    # Main document
      

    def _write_document(self) -> None:
        self._write(r"\begin{document}")
        self._write("")

        players_line = r"\enspace ".join(
            [f"P{i} {agent.name}" for i, agent in enumerate(self.agents)]
        )

        self._write(
            rf"\noindent\textbf{{Variant:}} {self.variant_name}\enspace"
            rf"\textbf{{Players:}} {players_line}"
        )
        self._write("")
        self._write(r"\vspace{0.10cm}\hrule\vspace{0.12cm}")
        self._write("")

        turn_data_list = [td for td in self.turn_states if td["turn"] > 0 and not td.get("final")]
        rounds = self._group_into_visual_rounds(turn_data_list)
        placements, round_infos = self._compute_layout(rounds)

        pages = self._paginate(round_infos)

        for page_idx, page_rounds in enumerate(pages):
            if page_idx > 0:
                self._write(r"\newpage")
                self._write("")

            page_offset_y = page_rounds[0]["current_top"]
            is_last_page = (page_idx == len(pages) - 1)

            self._write(r"\begin{center}")
            self._write(r"\resizebox{\linewidth}{!}{%")
            self._write(r"\begin{tikzpicture}")
            self._write("")

            self._write_headers()
            self._write_grid_rules(page_rounds, page_offset_y)
            self._write_round_labels(page_rounds, page_offset_y)
            self._write_turn_nodes(placements, page_rounds, page_offset_y)
            if is_last_page:
                self._write_empty_slots_for_last_incomplete_round(rounds, round_infos, page_offset_y)
            self._write_arrows(rounds, placements, page_rounds, page_offset_y)

            self._write("")
            self._write(r"\end{tikzpicture}%")
            self._write(r"}")
            self._write(r"\end{center}")
            self._write("")

        self._write_final_state(self.turn_states[-1]["state"])
        self._write("")
        self._write(r"\end{document}")

    def _write_preamble(self) -> None:
        self._write(r"""\documentclass[a4paper,10pt]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{xcolor}
\usepackage{tikz}
\usetikzlibrary{calc,positioning,shapes.geometric,arrows.meta}
\usepackage[a4paper,margin=1cm]{geometry}
\usepackage{graphicx}
\usepackage{etoolbox}

\setlength{\parindent}{0pt}
\setlength{\parskip}{0pt}

% =========================================================
% Card dimensions
% =========================================================
\def\CardWo{1.0}
\def\CardHo{1.5}
\def\CGapo{0.10}
\def\SuitPosX{0.50}
\def\SuitPosY{0.40}
\def\SuitScaleFactor{0.27}

% =========================================================
% SUIT SYMBOLS
% =========================================================
\newcommand{\SuitOros}[1]{%
  \shade[inner color=yellow!35!white,outer color=orange!60!yellow](0,0)circle(1.35);
  \shade[inner color=yellow!35!white,outer color=orange!50!yellow](0,0)circle(1.02);
  \draw[yellow!45!orange!80!black,line width={1.15*#1}](0,0)circle(1.35);
  \draw[yellow!45!orange!80!black,line width={0.90*#1}](0,0)circle(1.02);
  \begin{scope}[rotate=22.5]
    \foreach \a in{0,45,...,315}{
      \begin{scope}[rotate=\a]
        \fill[yellow!35!orange!90!black](0,0.48)--(0.10,0)--(0,-0.48)--(-0.10,0)--cycle;
      \end{scope}
    }
  \end{scope}
  \fill[yellow!35!orange!90!black](0,0)circle(0.10);
  \foreach \a in{0,30,...,330}{\fill[yellow!35!orange!90!black](\a:1.18)circle(0.05);}
}

\newcommand{\SuitCopas}[1]{%
  \begin{scope}[scale=0.74,yshift=0.35cm,line cap=round,line join=round]
    \filldraw[fill=red!80!black,draw=red!30!black,line width={1.00*#1}](-0.24,-1.45)rectangle(0.24,-2.25);
    \filldraw[fill=red!80!black,draw=red!30!black,line width={1.00*#1}](-0.70,-2.25)rectangle(0.70,-2.62);
    \filldraw[fill=red!80!black,draw=red!30!black,line width={1.10*#1}]
      (-1.25,0.75)..controls(-1.55,0.10)and(-1.40,-0.95)..(-0.55,-1.45)
      ..controls(-0.20,-1.65)and(0.20,-1.65)..(0.55,-1.45)
      ..controls(1.40,-0.95)and(1.55,0.10)..(1.25,0.75)
      ..controls(0.80,0.55)and(-0.80,0.55)..(-1.25,0.75)--cycle;
    \filldraw[fill=cyan!35!black,draw=black,line width={1.10*#1}](0,0.78)ellipse(1.24 and 0.36);
    \path[fill=white,opacity=0.45]
      (-0.98,0.25)..controls(-1.05,-0.15)and(-0.78,-0.85)..(-0.42,-1.15)
      ..controls(-0.18,-1.35)and(0.02,-1.28)..(0.08,-1.08)
      ..controls(-0.18,-1.02)and(-0.48,-0.75)..(-0.70,-0.35)
      ..controls(-0.88,-0.02)and(-0.95,0.20)..(-0.98,0.25)--cycle;
    \fill[yellow!80!orange](-0.40,-1.00)--(-0.22,-0.90)--(-0.22,-1.15)--(-0.40,-1.25)--cycle;
    \fill[yellow!80!orange](0.00,-0.92)--(0.13,-1.08)--(0.00,-1.32)--(-0.13,-1.08)--cycle;
    \fill[yellow!80!orange](0.40,-1.00)--(0.22,-0.90)--(0.22,-1.15)--(0.40,-1.25)--cycle;
  \end{scope}
}

\newcommand{\SuitEspadas}[1]{%
  \filldraw[fill=gray!20,draw=gray!70!black,line width={1.00*#1}]
    (0,1.50)..controls(0.58,0.72)and(0.48,-0.28)..(0,-1.14)
            ..controls(-0.48,-0.28)and(-0.58,0.72)..(0,1.50)--cycle;
  \draw[gray!60!black,line width={0.70*#1}](0,1.20)--(0,-0.95);
  \filldraw[fill=yellow!65!orange,draw=orange!70!black,line width={1.00*#1}](-0.92,-1.08)rectangle(0.92,-0.84);
  \filldraw[fill=yellow!65!orange,draw=orange!70!black,line width={1.00*#1}](-0.16,-1.48)rectangle(0.16,-1.08);
  \filldraw[fill=yellow!65!orange,draw=orange!70!black,line width={1.00*#1}](0,-1.68)circle(0.17);
}

\newcommand{\SuitBastos}[1]{%
  \begin{scope}[rotate=18]
    \shade[left color=brown!65!black,right color=brown!35!yellow](-0.20,-1.55)rectangle(0.20,1.20);
    \draw[brown!35!black,line width={1.00*#1}](-0.20,-1.55)rectangle(0.20,1.20);
    \foreach \yy in{-0.95,-0.35,0.25,0.82}{\draw[brown!25!black,line width={0.65*#1}](0,\yy)ellipse(0.12 and 0.06);}
    \shade[inner color=brown!35!yellow,outer color=brown!75!black](0,1.55)ellipse(0.66 and 0.44);
    \draw[brown!35!black,line width={1.00*#1}](0,1.55)ellipse(0.66 and 0.44);
    \filldraw[fill=green!25!black,draw=green!40!black,line width={0.50*#1}]
      (-0.20,0.45)..controls(-0.72,0.58)and(-0.70,0.14)..(-0.20,0.18)--cycle;
    \filldraw[fill=green!25!black,draw=green!40!black,line width={0.50*#1}]
      (0.20,-0.02)..controls(0.72,0.12)and(0.70,-0.28)..(0.20,-0.30)--cycle;
  \end{scope}
}

\newcommand{\SuitColor}[1]{%
  \ifcase#1\relax\or yellow!35!orange!80!black\or red!75!black\or gray!45!black\or brown!60!black\fi}
\newcommand{\DrawSuit}[2]{%
  \ifcase#1\relax\or\SuitOros{#2}\or\SuitCopas{#2}\or\SuitEspadas{#2}\or\SuitBastos{#2}\fi}
\newcommand{\RankValue}[1]{%
  \ifcase#1\relax\or 1\or 2\or 3\or 4\or 5\or 6\or 7\or 10\or 11\or 12\fi}

% =========================================================
% Card rendering
% =========================================================
\newcommand{\DrawCardScaled}[5]{%
  \begin{scope}[shift={(#1,#2)}]
    \pgfmathsetmacro{\CardW}{\CardWo*#5}
    \pgfmathsetmacro{\CardH}{\CardHo*#5}
    \pgfmathsetmacro{\S}{min(\CardW,\CardH)}
    \pgfmathsetmacro{\CardLW}{max(0.08,min(0.9,0.22*\S))}
    \pgfmathsetmacro{\CornerR}{max(0.015,min(0.12,0.06*\S))}
    \pgfmathsetmacro{\SuitLW}{max(0.06,min(0.35,0.18*\S))}
    \ifnum#3<8
      \pgfmathsetmacro{\NumPref}{11.5*\S}
    \else
      \pgfmathsetmacro{\NumPref}{9.2*\S}
    \fi
    \pgfmathsetmacro{\NumSizePt}{max(2.6,min(\NumPref,42.0))}
    \pgfmathsetmacro{\NumBasePt}{1.05*\NumSizePt}
    \ifnum#3<8
      \pgfmathsetmacro{\NumX}{\CardW*0.22}
    \else
      \pgfmathsetmacro{\NumX}{\CardW*0.29}
    \fi
    \pgfmathsetmacro{\NumY}{\CardH*0.87}
    \filldraw[fill=white,draw=black,rounded corners=\CornerR cm,line width=\CardLW pt]
      (0,0)rectangle(\CardW,\CardH);
    \node[anchor=center,text=\SuitColor{#4},inner sep=0pt] at(\NumX,\NumY)
      {\fontsize{\NumSizePt pt}{\NumBasePt pt}\selectfont\bfseries\RankValue{#3}};
    \begin{scope}[shift={({\CardW*\SuitPosX},{\CardH*\SuitPosY})},scale={\SuitScaleFactor*\S},transform shape]
      \DrawSuit{#4}{\SuitLW}
    \end{scope}
  \end{scope}
}

% =========================================================
% Layout
% =========================================================
\def\TurnW{7.00}
\def\CZero{2.20}
\def\COne{9.60}
\def\CTwo{17.00}
\def\CThree{24.40}

\tikzset{
  TurnBox/.style  ={draw=black!60,rounded corners=2pt,line width=0.65pt,fill=white},
  InnerBox/.style ={draw=black!30,rounded corners=1.5pt,line width=0.45pt,fill=white},
  TinyLbl/.style  ={font=\tiny,text=black!60},
  ColHdr/.style   ={font=\scriptsize\bfseries,text=white,fill=black!72,
                    rounded corners=2.5pt,inner sep=2.5pt,
                    minimum width=\TurnW cm,minimum height=0.50cm,anchor=center},
  RndLbl/.style   ={font=\fontsize{6.5pt}{8pt}\selectfont\bfseries,
                    text=black!80,anchor=east,inner sep=1.5pt},
  RndSub/.style   ={font=\fontsize{5.5pt}{6.5pt}\selectfont\itshape,
                    text=black!50,anchor=east,inner sep=1.5pt},
  EmptySlot/.style={draw=black!18,dashed,rounded corners=2pt,
                    line width=0.45pt,fill=black!02},
  FlowArrow/.style={-{Stealth[length=2.5mm,width=1.8mm]},line width=0.75pt,
                    blue!60!black,shorten >=1.5pt,shorten <=1.5pt},
  WrapArrow/.style={-{Stealth[length=2.5mm,width=1.8mm]},line width=0.75pt,
                    blue!45!black,shorten >=1.5pt,shorten <=1.5pt,densely dashed},
}

% =========================================================
% Adaptive turncard pic
% widened action boxes
% =========================================================
\tikzset{
  pics/turncard/.style args={#1/#2/#3/#4}{
    code={
      \def\W{\TurnW}

      \def\lmarg{0.10}
      \def\gap{0.14}
      \def\moveW{1.55}

      \ifstrequal{#3}{Play}{\def\moveW{1.55}}{}
      \ifstrequal{#3}{Pass}{\def\moveW{1.75}}{}
      \ifstrequal{#3}{RollDice}{\def\moveW{1.65}}{}

      \pgfmathsetmacro{\handW}{\W-\lmarg-\gap-\moveW-0.08}
      \pgfmathsetmacro{\effw}{\handW-0.08}
      \pgfmathsetmacro{\Nc}{#1}
      \pgfmathsetmacro{\rawW}{\Nc + (\Nc-1)*\CGapo}
      \pgfmathsetmacro{\cs}{min(1.0, \effw/\rawW)}
      \pgfmathsetmacro{\cw}{\CardWo*\cs}
      \pgfmathsetmacro{\ch}{\CardHo*\cs}
      \pgfmathsetmacro{\cg}{\CGapo*\cs}

      \pgfmathsetmacro{\boxTop}{0.02}
      \pgfmathsetmacro{\topY}{-0.01}
      \pgfmathsetmacro{\pad}{0.04}
      \pgfmathsetmacro{\boxBot}{-0.17-\ch}
      \pgfmathsetmacro{\boxMid}{(\boxTop+\boxBot)/2}

      \pgfmathsetmacro{\cardY}{\topY-\pad-\ch}
      \pgfmathsetmacro{\moveCX}{\lmarg+\handW+\gap+\moveW/2}
      \pgfmathsetmacro{\moveCY}{\boxMid}

      \draw[TurnBox](0,\boxTop)rectangle(\W,\boxBot);

      \foreach \r/\s/\h [count=\i from 0] in {#2}{
        \pgfmathsetmacro{\cardX}{\lmarg+\pad+\i*(\cw+\cg)}
        \DrawCardScaled{\cardX}{\cardY}{\r}{\s}{\cs}
        \ifnum\h=1
          \draw[green!65!black,line width=0.95pt]
            (\cardX,\cardY)rectangle(\cardX+\cw,\cardY+\ch);
        \fi
      }

      \ifstrequal{#3}{Play}{
        \pgfmathsetmacro{\aboxW}{1.45}
        \pgfmathsetmacro{\aboxH}{0.56}
        \draw[InnerBox]
          (\moveCX-\aboxW/2,\moveCY+\aboxH/2)
          rectangle
          (\moveCX+\aboxW/2,\moveCY-\aboxH/2);
        \node[TinyLbl,text width=1.33cm,align=center,anchor=center]
          at(\moveCX,\moveCY){\textbf{#3}\\[-0.02cm]#4};
      }{
        \ifstrequal{#3}{Pass}{
          \pgfmathsetmacro{\aboxW}{1.62}
          \pgfmathsetmacro{\aboxH}{0.58}
          \draw[InnerBox]
            (\moveCX-\aboxW/2,\moveCY+\aboxH/2)
            rectangle
            (\moveCX+\aboxW/2,\moveCY-\aboxH/2);
          \node[TinyLbl,text width=1.48cm,align=center,anchor=center]
            at(\moveCX,\moveCY){\textbf{#3}\\[-0.02cm]#4};
        }{
          \pgfmathsetmacro{\aboxW}{1.48}
          \pgfmathsetmacro{\aboxH}{0.56}
          \draw[InnerBox]
            (\moveCX-\aboxW/2,\moveCY+\aboxH/2)
            rectangle
            (\moveCX+\aboxW/2,\moveCY-\aboxH/2);
          \node[TinyLbl,text width=1.34cm,align=center,anchor=center]
            at(\moveCX,\moveCY){\textbf{#3}\\[-0.02cm]#4};
        }
      }

      \coordinate(-top)  at(\W/2,\boxTop);
      \coordinate(-bot)  at(\W/2,\boxBot);
      \coordinate(-left) at(0,\boxMid);
      \coordinate(-right)at(\W,\boxMid);
    }
  }
}

\newcommand{\PlaceTurnAt}[4]{%
  \path (#1,#2) pic[anchor=north west,name=#4]{turncard={#3}};
}

\newcommand{\LinkDown}[2]{\draw[FlowArrow](#1-bot)--(#2-top);}
\newcommand{\LinkRight}[2]{\draw[FlowArrow](#1-right)--(#2-left);}
\newcommand{\LinkRightUp}[2]{\draw[FlowArrow](#1-right)--++(0.20,0)|-(#2-left);}

% #3 = absolute Y coordinate of the separator line
\newcommand{\LinkWrapAt}[3]{%
  \draw[WrapArrow]
    (#1-bot)
    -- (#1-bot |- 0,#3)
    -| ([yshift=0.18cm]#2-top)
    -- (#2-top);
}
""")

    def _write_postamble(self) -> None:
        pass

      
    # Layout
      

    def _group_into_visual_rounds(self, turn_data_list: List[dict]) -> List[List[dict]]:
        """
        A visual round starts when player order wraps around:
        e.g. ... P3 -> P0 starts a new round block.
        Repeated same-player actions (roll then play, double play) stay in the same round.
        """
        if not turn_data_list:
            return []

        rounds: List[List[dict]] = []
        current_round: List[dict] = []
        prev_player: Optional[int] = None

        for td in turn_data_list:
            player = td["state"].current_player

            if prev_player is not None and player < prev_player:
                rounds.append(current_round)
                current_round = []

            current_round.append(td)
            prev_player = player

        if current_round:
            rounds.append(current_round)

        return rounds

    def _move_width(self, move) -> float:
        if isinstance(move, PlayCard):
            return 1.55
        if isinstance(move, Pass):
            return 1.75
        if isinstance(move, RollDice):
            return 1.65
        return 1.55

    def _box_height(self, td: dict) -> float:
        """
        Exact outer-box height induced by the LaTeX pic macros.
        boxTop = 0.02, boxBot = -0.17 - ch, where ch = CardHo * cs.
        height = 0.19 + ch.
        """
        nc = max(1, td["state"].get_current_player().hand_size())
        move_w = self._move_width(td["move"])

        lmarg = 0.10
        gap = 0.14
        hand_w = self.TURN_W - lmarg - gap - move_w - 0.08
        effw = hand_w - 0.08
        raww = nc + (nc - 1) * 0.10
        cs = min(1.0, effw / raww) if raww > 0 else 1.0
        ch = 1.5 * cs
        return 0.19 + ch

    def _compute_layout(self, rounds: List[List[dict]]) -> Tuple[Dict[str, dict], List[dict]]:
        placements: Dict[str, dict] = {}
        round_infos: List[dict] = []

        current_top = 0.0
        global_turn_index = 1

        for round_idx, round_turns in enumerate(rounds, start=1):
            # One contiguous stack per player within the round.
            by_player: Dict[int, List[Tuple[str, dict]]] = {0: [], 1: [], 2: [], 3: []}
            chronological_groups: List[Tuple[int, str, dict]] = []

            for td in round_turns:
                tid = f"T{global_turn_index}"
                player = td["state"].current_player
                by_player[player].append((tid, td))
                if not chronological_groups or chronological_groups[-1][0] != player:
                    chronological_groups.append((player, tid, td))
                global_turn_index += 1

            stack_heights: Dict[int, float] = {}
            for p in range(4):
                stack = by_player[p]
                if not stack:
                    stack_heights[p] = 0.0
                else:
                    heights = [self._box_height(td) for _, td in stack]
                    stack_heights[p] = sum(heights) + self.INTRA_GAP * max(0, len(heights) - 1)

            round_height = max(stack_heights.values()) if stack_heights else 0.0
            round_center_y = current_top - round_height / 2.0
            round_bottom = current_top - round_height

            for p in range(4):
                stack = by_player[p]
                if not stack:
                    continue

                stack_height = stack_heights[p]
                stack_top = current_top - (round_height - stack_height) / 2.0
                y_cursor = stack_top

                for tid, td in stack:
                    h = self._box_height(td)
                    placements[tid] = {
                        "tid": tid,
                        "td": td,
                        "player": p,
                        "x": self.COL_X[p],
                        "y": y_cursor,
                        "height": h,
                        "round": round_idx,
                    }
                    y_cursor -= (h + self.INTRA_GAP)

            round_infos.append({
                "round_idx": round_idx,
                "turns": round_turns,
                "turn_ids": [f"T{global_turn_index - len(round_turns) + i}" for i in range(len(round_turns))],
                "current_top": current_top,
                "height": round_height,
                "center_y": round_center_y,
                "bottom": round_bottom,
                "groups": chronological_groups,
                "by_player": by_player,
            })

            current_top = round_bottom - self.INTER_ROUND_GAP

        return placements, round_infos

      
    # TikZ writing
      

    def _paginate(self, round_infos: List[dict], budget: float = 40) -> List[List[dict]]:
        """Split round_infos into pages so each page's content height ≤ budget cm."""
        pages: List[List[dict]] = []
        current_page: List[dict] = []
        page_start_y: Optional[float] = None

        for ri in round_infos:
            if page_start_y is None:
                page_start_y = ri["current_top"]

            # How far down from the page start does this round's bottom reach?
            used = page_start_y - ri["bottom"]  # positive number

            if current_page and used > budget:
                pages.append(current_page)
                current_page = [ri]
                page_start_y = ri["current_top"]
            else:
                current_page.append(ri)

        if current_page:
            pages.append(current_page)

        return pages

    def _write_headers(self) -> None:
        names = [agent.name for agent in self.agents]
        self._write(f"\\node[ColHdr] at (\\CZero +\\TurnW/2,0.72) {{P0 — {names[0]}}};")
        self._write(f"\\node[ColHdr] at (\\COne  +\\TurnW/2,0.72) {{P1 — {names[1]}}};")
        self._write(f"\\node[ColHdr] at (\\CTwo  +\\TurnW/2,0.72) {{P2 — {names[2]}}};")
        self._write(f"\\node[ColHdr] at (\\CThree+\\TurnW/2,0.72) {{P3 — {names[3]}}};")
        self._write("")

    def _write_grid_rules(self, round_infos: List[dict], page_offset_y: float = 0.0) -> None:
        def ly(y: float) -> float:
            return y - page_offset_y

        min_y = -2.0
        if round_infos:
            min_y = ly(min(info["bottom"] for info in round_infos)) - 0.45

        self._write(f"\\draw[black!22,line width=0.4pt](\\CZero,0.92)--(\\CZero,{min_y:.4f});")
        self._write(f"\\draw[black!12,line width=0.3pt](\\COne -0.20,0.92)--(\\COne -0.20,{min_y:.4f});")
        self._write(f"\\draw[black!12,line width=0.3pt](\\CTwo -0.20,0.92)--(\\CTwo -0.20,{min_y:.4f});")
        self._write(f"\\draw[black!12,line width=0.3pt](\\CThree-0.20,0.92)--(\\CThree-0.20,{min_y:.4f});")

        # Draw a separator line after every round (including the last one on the page)
        for info in round_infos:
            sep_y = ly(info["bottom"] - self.INTER_ROUND_GAP / 2.0)
            self._write(f"\\draw[black!30,line width=0.55pt](0.10,{sep_y:.4f})--(31.30,{sep_y:.4f});")

        self._write("")

    def _write_round_labels(self, round_infos: List[dict], page_offset_y: float = 0.0) -> None:
        def ly(y: float) -> float:
            return y - page_offset_y

        for info in round_infos:
            self._write(f"\\node[RndLbl] at (2.10,{ly(info['center_y']):.4f}) {{Round {info['round_idx']}}};")

        if round_infos:
            last = round_infos[-1]
            active_players = {td["state"].current_player for td in last["turns"]}
            if len(active_players) < 4:
                sub_y = ly(last["center_y"]) - 0.35
                self._write(f"\\node[RndSub] at (2.10,{sub_y:.4f}) {{\\textit{{(incomplete)}}}};")

        self._write("")

    def _format_cards_payload(self, td: dict) -> str:
        state = td["state"]
        hand = state.get_current_player().hand
        legal_cards = {m.card for m in td["legal_moves"] if isinstance(m, PlayCard)}

        sorted_hand = sorted(hand, key=lambda c: (self.SUIT_TO_LATEX[c.suit], self.RANK_TO_INDEX[c.rank]))
        parts = []
        for card in sorted_hand:
            r = self.RANK_TO_INDEX[card.rank]
            s = self.SUIT_TO_LATEX[card.suit]
            h = 1 if card in legal_cards else 0
            parts.append(f"{r}/{s}/{h}")
        return ",".join(parts)

    def _format_move_payload(self, td: dict) -> Tuple[str, str]:
        move = td["move"]
        pre_state = td["state"]
        post_state = td["post_state"]

        if isinstance(move, PlayCard):
            return ("Play", f"{self.SUIT_NAMES[move.card.suit]} {move.card.rank}")

        if isinstance(move, Pass):
            if move.voluntary:
                return ("Pass", f"(voluntary, -{pre_state.variant.voluntary_pass_penalty}pt)")
            return ("Pass", "(forced)")

        if isinstance(move, RollDice):
            return ("RollDice", self._infer_roll_detail(pre_state, post_state))

        return ("Move", "...")

    def _infer_roll_detail(self, old_state: GameState, new_state: GameState) -> str:
        cur = old_state.current_player
        old_player = old_state.players[cur]
        new_player = new_state.players[cur]

        if new_state.dice_state.wild_active and not old_state.dice_state.wild_active:
            return r"{\color{green!60!black}WILD}"

        if new_state.dice_state.double_play_active and not old_state.dice_state.double_play_active:
            return r"{\color{green!60!black}DOUBLE}"

        # GIVE_CARD
        if new_player.hand_size() < old_player.hand_size():
            for i, p in enumerate(new_state.players):
                if i != cur and p.hand_size() > old_state.players[i].hand_size():
                    return rf"{{\color{{green!60!black}}GIVE P{i}}}"
            return r"{\color{green!60!black}GIVE}"

        # TAKE_CARDS
        if new_player.hand_size() > old_player.hand_size():
            for i, p in enumerate(new_state.players):
                if i != cur and p.hand_size() < old_state.players[i].hand_size():
                    return rf"{{\color{{red!55!black}}TAKE P{i}}}"
            return r"{\color{red!55!black}TAKE}"

        # INFO_REVEAL
        old_target = old_state.dice_state.get_revealed_target(cur)
        new_target = new_state.dice_state.get_revealed_target(cur)
        if old_target is None and new_target is not None:
            return rf"{{\color{{green!60!black}}SEE P{new_target}}}"

        # REVEAL_HAND
        viewers = [
            i for i in range(len(new_state.players))
            if i != cur and new_state.dice_state.get_revealed_target(i) == cur
        ]
        if viewers:
            return r"{\color{red!55!black}REVEALED}"

        # NEGATIVE_POINTS
        if new_player.match_score < old_player.match_score:
            delta = old_player.match_score - new_player.match_score
            return rf"{{\color{{red!55!black}}-{delta} pts}}"

        # FORCED_PASS / skip
        if new_state.current_player != old_state.current_player:
            return r"{\color{red!55!black}SKIP}"

        return "..."

    def _write_turn_nodes(self, placements: Dict[str, dict], page_rounds: List[dict], page_offset_y: float = 0.0) -> None:
        # Collect turn IDs that belong to this page
        page_tids: set = set()
        for ri in page_rounds:
            for tid in ri["turn_ids"]:
                page_tids.add(tid)

        for turn_num in range(1, len(placements) + 1):
            tid = f"T{turn_num}"
            if tid not in page_tids:
                continue
            info = placements[tid]
            td = info["td"]
            x = info["x"]
            y = info["y"] - page_offset_y

            nc = td["state"].get_current_player().hand_size()
            cards = self._format_cards_payload(td)
            action, detail = self._format_move_payload(td)

            self._write(
                f"\\PlaceTurnAt{{{x:.4f}}}{{{y:.4f}}}%\n"
                f"  {{{nc}/{{{cards}}}%\n"
                f"   /{action}/{{{detail}}}}}{{{tid}}}"
            )

        self._write("")

    def _write_empty_slots_for_last_incomplete_round(self, rounds: List[List[dict]], round_infos: List[dict], page_offset_y: float = 0.0) -> None:
        if not rounds or not round_infos:
            return

        last_round = rounds[-1]
        info = round_infos[-1]
        active_players = {td["state"].current_player for td in last_round}

        if len(active_players) == 4:
            return

        top = info["current_top"] - page_offset_y
        bottom = info["bottom"] - page_offset_y
        center_y = info["center_y"] - page_offset_y

        for p in range(4):
            if p in active_players:
                continue
            x = self.COL_X[p]
            self._write(f"\\draw[EmptySlot]({x:.4f},{top:.4f})rectangle({x + self.TURN_W:.4f},{bottom:.4f});")
            self._write(
                f"\\node[font=\\fontsize{{5pt}}{{6pt}}\\selectfont,text=black!30,anchor=center] "
                f"at({x + self.TURN_W/2:.4f},{center_y:.4f}){{\\textit{{-- did not play --}}}};"
            )

        self._write("")

    def _write_arrows(self, rounds: List[List[dict]], placements: Dict[str, dict], page_rounds: List[dict], page_offset_y: float = 0.0) -> None:
        # Build round_turn_names only for rounds on this page
        page_round_turn_names: List[List[str]] = []
        for ri in page_rounds:
            page_round_turn_names.append(list(ri["turn_ids"]))

        # inside-round arrows
        for names in page_round_turn_names:
            # vertical arrows for same-player consecutive turns
            for i in range(len(names) - 1):
                a = placements[names[i]]
                b = placements[names[i + 1]]
                if a["player"] == b["player"]:
                    self._write(f"\\LinkDown{{{names[i]}}}{{{names[i + 1]}}}")

            # arrows between consecutive player groups
            groups: List[List[str]] = []
            current_group = [names[0]]
            for i in range(1, len(names)):
                if placements[names[i]]["player"] == placements[names[i - 1]]["player"]:
                    current_group.append(names[i])
                else:
                    groups.append(current_group)
                    current_group = [names[i]]
            groups.append(current_group)

            for i in range(len(groups) - 1):
                a = groups[i][-1]
                b = groups[i + 1][0]
                ya = placements[a]["y"]
                yb = placements[b]["y"]
                if abs(ya - yb) < 1e-6:
                    self._write(f"\\LinkRight{{{a}}}{{{b}}}")
                else:
                    self._write(f"\\LinkRightUp{{{a}}}{{{b}}}")

        # wrap arrows between rounds (only within this page)
        for i in range(len(page_rounds) - 1):
            ri = page_rounds[i]
            last_turn = page_round_turn_names[i][-1]
            next_turn = page_round_turn_names[i + 1][0]
            # Absolute page-local separator Y (the horizontal rule between these rounds)
            sep_y = ri["bottom"] - self.INTER_ROUND_GAP / 2.0 - page_offset_y
            self._write(f"\\LinkWrapAt{{{last_turn}}}{{{next_turn}}}{{{sep_y:.4f}}}")

        self._write("")

      
    # Final state
      

    def _write_final_state(self, state: GameState) -> None:
        winner_text = "None"
        if state.winner is not None:
            winner_text = f"P{state.winner} ({self.agents[state.winner].name})"

        self._write(r"\vspace{0.20cm}")
        self._write(rf"\noindent\textbf{{\large Game Over}}\quad\textbf{{Winner:}} {winner_text}")
        self._write("")
        self._write(r"\vspace{0.18cm}")
        self._write(r"\begin{center}")
        self._write(r"\begin{tabular}{|c|l|c|}")
        self._write(r"\hline")
        self._write(r"\textbf{P} & \textbf{Agent} & \textbf{Match}\\\hline")

        for i, player in enumerate(state.players):
            star = r"$\star$" if i == state.winner else ""
            self._write(
                f"{i}{star} & {self.agents[i].name} & {player.match_score:+d}\\\\\\hline"
            )

        self._write(r"\end{tabular}")
        self._write(r"\end{center}")


  
# CLI
  

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate one fixed-grid LaTeX visualization for one variant."
    )
    parser.add_argument(
        "variant",
        nargs="?",
        default="baseline",
        help="Variant name. Default: baseline",
    )
    parser.add_argument(
        "--agents",
        nargs=4,
        metavar=("A1", "A2", "A3", "A4"),
        default=DEFAULT_AGENT_KEYS,
        help="Exactly 4 agents. Default: mcts-deep rl balanced aggressive",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output .tex file",
    )
    parser.add_argument(
        "--list-variants",
        action="store_true",
        help="List available variants and exit",
    )
    parser.add_argument(
        "--list-agents",
        action="store_true",
        help="List available agents and exit",
    )
    return parser


def print_variants() -> None:
    print("Available variants:")
    for spec in VARIANT_SPECS:
        print(f"  - {spec.name}")


def print_agents() -> None:
    print("Available agents:")
    for name in sorted(AGENT_BUILDERS.keys()):
        print(f"  - {name}")


def resolve_variant(name: str) -> Tuple[str, VariantConfig]:
    key = slugify(name)
    if key not in VARIANT_REGISTRY:
        raise ValueError(f"Unknown variant '{name}'. Use --list-variants.")
    spec = VARIANT_REGISTRY[key]
    return spec.name, build_variant(spec)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.list_variants:
        print_variants()
        return

    if args.list_agents:
        print_agents()
        return

    variant_name, variant = resolve_variant(args.variant)
    agents = build_agents(args.agents)

    output_file = args.output
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join("output", "flows")
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"{slugify(variant_name)}_{timestamp}.tex")
    else:
        output_parent = os.path.dirname(output_file)
        if output_parent:
            os.makedirs(output_parent, exist_ok=True)

    visualizer = LaTeXGridVisualizer(
        agents=agents,
        variant=variant,
        variant_name=variant_name,
        output_file=output_file,
    )
    visualizer.play_and_visualize()


if __name__ == "__main__":
    main()