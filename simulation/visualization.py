#!/usr/bin/env python3
"""
LaTeX/TikZ visualization tool for Cinquillo 2.0.
Flow-based layout: turn containers placed side-by-side with automatic wrapping and arrows.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List, TextIO, Tuple
from game.entities import GameState, VariantConfig, Card, Suit
from game.rules import Rules, PlayCard, RollDice, Pass
from agents.mcts_agent import MCTSAgentDeep, MCTSAgentStandard
from agents.rl_agent import RLAgent
from agents.base_agents import Agent, create_balanced_heuristic, create_aggressive_heuristic, create_defensive_heuristic, RandomAgent


class LaTeXGameVisualizer:
    """Generates flow-based LaTeX/TikZ visualizations with adaptive containers."""
    
    SUIT_TO_LATEX = {
        Suit.OROS: 1,
        Suit.COPAS: 2,
        Suit.ESPADAS: 3,
        Suit.BASTOS: 4
    }
    
    SUIT_NAMES = {
        Suit.OROS: "Oros",
        Suit.COPAS: "Copas",
        Suit.ESPADAS: "Espadas",
        Suit.BASTOS: "Bastos"
    }
    
    RANK_TO_INDEX = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 10: 8, 11: 9, 12: 10}
    
    def __init__(self, agents: List[Agent], variant: VariantConfig, output_file: str = None):
        self.agents = agents
        self.variant = variant
        self.num_players = len(agents)
        self.output_file = output_file
        self.file_handle = None
        self.turn_states = []
    
    def play_and_visualize(self):
        """Play game and generate LaTeX visualization."""
        state = Rules.initialize_game(self.num_players, self.variant)
        
        turn_count = 0
        max_turns = 1000
        
        self.turn_states.append({
            'turn': 0,
            'state': state.copy(),
            'agent': None,
            'move': None,
            'legal_moves': [],
            'post_state': None
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
                'turn': turn_count,
                'state': pre_state,
                'post_state': state.copy(),
                'agent': agent,
                'move': chosen_move,
                'legal_moves': legal_moves
            })
        
        Rules.compute_round_scores(state)
        self.turn_states.append({
            'turn': turn_count + 1,
            'state': state,
            'agent': None,
            'move': None,
            'legal_moves': [],
            'post_state': None,
            'final': True
        })
        
        self._generate_latex()
    
    def _generate_latex(self):
        """Generate complete LaTeX document."""
        if self.output_file:
            self.file_handle = open(self.output_file, 'w', encoding='utf-8')
        
        try:
            self._write_preamble()
            self._write_initial_state()
            
            # Write turns in flow layout
            turn_data_list = [td for td in self.turn_states if td['turn'] > 0 and not td.get('final')]
            self._write_turns_flow(turn_data_list)
            
            self._write(r"\clearpage")
            
            final_state = self.turn_states[-1]
            self._write_final_state(final_state)
            
            self._write_postamble()
            
            if self.output_file:
                print(f"✓ LaTeX visualization saved to: {self.output_file}")
                print(f"  Compile with: pdflatex {self.output_file}")
        finally:
            if self.file_handle:
                self.file_handle.close()
    
    def _write(self, text: str = ""):
        """Write to file or stdout."""
        if self.file_handle:
            self.file_handle.write(text + "\n")
        else:
            print(text)
    
    def _calculate_turn_width(self, turn_data) -> Tuple[float, float, float]:
        """Calculate width of turn container components.
        Returns: (hand_width, move_width, total_width) in cm
        """
        state = turn_data['state']
        move = turn_data['move']
        player = state.get_current_player()
        hand_size = player.hand_size()
        
        # Standard dimensions
        card_w = 1.0
        card_gap = 0.10
        max_row_w = 10.9
        
        # Calculate hand width with scaling
        raw_w = hand_size * card_w + (hand_size - 1) * card_gap if hand_size > 0 else 0
        scale = min(1.0, max_row_w / raw_w) if raw_w > 0 else 1.0
        scaled_row_w = raw_w * scale
        hand_w = scaled_row_w + 0.5  # Add padding
        
        # Calculate move width based on move type
        if isinstance(move, PlayCard):
            move_w = 1.8  # Just one card + padding
        elif isinstance(move, RollDice):
            move_w = 3.2  # Text: "RollDice\nTAKE from P1"
        elif isinstance(move, Pass):
            move_w = 2.4  # Text: "Pass\n(voluntary, -1pt)"
        else:
            move_w = 1.8
        
        # Total turn width: hand + gap + move + outer padding
        gap = 0.3
        outer_pad = 0.4
        total_w = hand_w + gap + move_w + 2 * outer_pad
        
        return (hand_w, move_w, total_w)
    
    def _write_preamble(self):
        """Write LaTeX preamble."""
        self._write(r"""\documentclass[a4paper,10pt]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage[spanish]{babel}
\usepackage{xcolor}
\usepackage{tikz}
\usetikzlibrary{calc,positioning,shapes.geometric,arrows.meta}
\usepackage[margin=1cm]{geometry}

\setlength{\parindent}{0pt}
\setlength{\parskip}{0.3em}

% Standard card dimensions
\def\CardWo{1.0}
\def\CardHo{1.5}
\def\CGapo{0.10}
\def\MaxRowW{10.9}

\def\SuitPosX{0.50}
\def\SuitPosY{0.40}
\def\SuitScaleFactor{0.27}

% =========================================================
% SUIT SYMBOLS
% =========================================================

% ---------- OROS ----------
\newcommand{\SuitOros}[1]{%
  \shade[inner color=yellow!35!white, outer color=orange!60!yellow] (0,0) circle (1.35);
  \shade[inner color=yellow!35!white, outer color=orange!50!yellow] (0,0) circle (1.02);
  \draw[yellow!45!orange!80!black, line width={1.15*#1}] (0,0) circle (1.35);
  \draw[yellow!45!orange!80!black, line width={0.90*#1}] (0,0) circle (1.02);
  \begin{scope}[rotate=22.5]
    \foreach \a in {0,45,...,315}{
      \begin{scope}[rotate=\a]
        \fill[yellow!35!orange!90!black] (0,0.48) -- (0.10,0) -- (0,-0.48) -- (-0.10,0) -- cycle;
      \end{scope}
    }
  \end{scope}
  \fill[yellow!35!orange!90!black] (0,0) circle (0.10);
  \foreach \a in {0,30,...,330}{
    \fill[yellow!35!orange!90!black] (\a:1.18) circle (0.05);
  }
}

% ---------- COPAS ----------
\newcommand{\SuitCopas}[1]{%
  \begin{scope}[scale=0.74, yshift=0.35cm, line cap=round, line join=round]
    \filldraw[fill=red!80!black, draw=red!30!black, line width={1.00*#1}]
      (-0.24,-1.45) rectangle (0.24,-2.25);
    \filldraw[fill=red!80!black, draw=red!30!black, line width={1.00*#1}]
      (-0.70,-2.25) rectangle (0.70,-2.62);
    \filldraw[fill=red!80!black, draw=red!30!black, line width={1.10*#1}]
      (-1.25,0.75) .. controls (-1.55,0.10) and (-1.40,-0.95) .. (-0.55,-1.45)
        .. controls (-0.20,-1.65) and (0.20,-1.65) .. (0.55,-1.45)
        .. controls (1.40,-0.95) and (1.55,0.10) .. (1.25,0.75)
        .. controls (0.80,0.55) and (-0.80,0.55) .. (-1.25,0.75) -- cycle;
    \filldraw[fill=cyan!35!black, draw=black, line width={1.10*#1}]
      (0,0.78) ellipse (1.24 and 0.36);
    \path[fill=white, opacity=0.45]
      (-0.98,0.25) .. controls (-1.05,-0.15) and (-0.78,-0.85) .. (-0.42,-1.15)
        .. controls (-0.18,-1.35) and (0.02,-1.28) .. (0.08,-1.08)
        .. controls (-0.18,-1.02) and (-0.48,-0.75) .. (-0.70,-0.35)
        .. controls (-0.88,-0.02) and (-0.95,0.20) .. (-0.98,0.25) -- cycle;
    \fill[yellow!80!orange] (-0.40,-1.00) -- (-0.22,-0.90) -- (-0.22,-1.15) -- (-0.40,-1.25) -- cycle;
    \fill[yellow!80!orange] ( 0.00,-0.92) -- ( 0.13,-1.08) -- ( 0.00,-1.32) -- (-0.13,-1.08) -- cycle;
    \fill[yellow!80!orange] ( 0.40,-1.00) -- ( 0.22,-0.90) -- ( 0.22,-1.15) -- ( 0.40,-1.25) -- cycle;
  \end{scope}
}

% ---------- ESPADAS ----------
\newcommand{\SuitEspadas}[1]{%
  \filldraw[fill=gray!20, draw=gray!70!black, line width={1.00*#1}]
    (0,1.50) .. controls (0.58,0.72) and (0.48,-0.28) .. (0,-1.14)
            .. controls (-0.48,-0.28) and (-0.58,0.72) .. (0,1.50) -- cycle;
  \draw[gray!60!black, line width={0.70*#1}] (0,1.20) -- (0,-0.95);
  \filldraw[fill=yellow!65!orange, draw=orange!70!black, line width={1.00*#1}]
    (-0.92,-1.08) rectangle (0.92,-0.84);
  \filldraw[fill=yellow!65!orange, draw=orange!70!black, line width={1.00*#1}]
    (-0.16,-1.48) rectangle (0.16,-1.08);
  \filldraw[fill=yellow!65!orange, draw=orange!70!black, line width={1.00*#1}]
    (0,-1.68) circle (0.17);
}

% ---------- BASTOS ----------
\newcommand{\SuitBastos}[1]{%
  \begin{scope}[rotate=18]
    \shade[left color=brown!65!black, right color=brown!35!yellow]
      (-0.20,-1.55) rectangle (0.20,1.20);
    \draw[brown!35!black, line width={1.00*#1}]
      (-0.20,-1.55) rectangle (0.20,1.20);
    \foreach \yy in {-0.95,-0.35,0.25,0.82}{
      \draw[brown!25!black, line width={0.65*#1}] (0,\yy) ellipse (0.12 and 0.06);
    }
    \shade[inner color=brown!35!yellow, outer color=brown!75!black]
      (0,1.55) ellipse (0.66 and 0.44);
    \draw[brown!35!black, line width={1.00*#1}] (0,1.55) ellipse (0.66 and 0.44);
    \filldraw[fill=green!25!black, draw=green!40!black, line width={0.50*#1}]
      (-0.20,0.45) .. controls (-0.72,0.58) and (-0.70,0.14) .. (-0.20,0.18) -- cycle;
    \filldraw[fill=green!25!black, draw=green!40!black, line width={0.50*#1}]
      (0.20,-0.02) .. controls (0.72,0.12) and (0.70,-0.28) .. (0.20,-0.30) -- cycle;
  \end{scope}
}

\newcommand{\SuitColor}[1]{%
  \ifcase#1\relax
  \or yellow!35!orange!80!black%
  \or red!75!black%
  \or gray!45!black%
  \or brown!60!black%
  \fi
}

\newcommand{\DrawSuit}[2]{%
  \ifcase#1\relax
  \or \SuitOros{#2}%
  \or \SuitCopas{#2}%
  \or \SuitEspadas{#2}%
  \or \SuitBastos{#2}%
  \fi
}

\newcommand{\RankValue}[1]{%
  \ifcase#1\relax
  \or 1\or 2\or 3\or 4\or 5\or 6\or 7\or 10\or 11\or 12%
  \fi
}

\newcommand{\DrawCardScaled}[5]{%
  \begin{scope}[shift={(#1,#2)}]
    \pgfmathsetmacro{\CardW}{\CardWo * #5}
    \pgfmathsetmacro{\CardH}{\CardHo * #5}
    \pgfmathsetmacro{\S}{min(\CardW,\CardH)}
    \pgfmathsetmacro{\CardLW}{max(0.08, min(0.9, 0.22*\S))}
    \pgfmathsetmacro{\CornerR}{max(0.015, min(0.12, 0.06*\S))}
    \pgfmathsetmacro{\SuitLW}{max(0.06, min(0.35, 0.18*\S))}
    
    \ifnum#3<8
      \pgfmathsetmacro{\NumPref}{11.5*\S}
    \else
      \pgfmathsetmacro{\NumPref}{9.2*\S}
    \fi
    \pgfmathsetmacro{\NumSizePt}{max(2.6, min(\NumPref,42.0))}
    \pgfmathsetmacro{\NumBasePt}{1.05*\NumSizePt}
    
    \ifnum#3<8
      \pgfmathsetmacro{\NumX}{\CardW*0.22}
    \else
      \pgfmathsetmacro{\NumX}{\CardW*0.29}
    \fi
    \pgfmathsetmacro{\NumY}{\CardH*0.87}
    
    \filldraw[fill=white, draw=black, rounded corners=\CornerR cm, line width=\CardLW pt]
      (0,0) rectangle (\CardW,\CardH);
    
    \node[anchor=center, text=\SuitColor{#4}, inner sep=0pt] at (\NumX,\NumY) {%
      \fontsize{\NumSizePt pt}{\NumBasePt pt}\selectfont\bfseries \RankValue{#3}%
    };
    
    \begin{scope}[shift={({\CardW*\SuitPosX},{\CardH*\SuitPosY})}, scale={\SuitScaleFactor*\S}, transform shape]
      \DrawSuit{#4}{\SuitLW}
    \end{scope}
  \end{scope}
}

\begin{document}

\title{Cinquillo 2.0 -- Game Flow Visualization}
\author{}
\date{}
\maketitle

""")
    
    def _write_postamble(self):
        """Write LaTeX document end."""
        self._write(r"\end{document}")
    
    def _write_initial_state(self):
        """Write initial game state."""
        state = self.turn_states[0]['state']
        variant = state.variant
        
        self._write(r"\noindent\textbf{Variant:} " + ("Winner Takes All" if variant.scoring_mode.value == "winner_takes_all" else "Double Penalty"))
        self._write(r"\quad\textbf{Players:} " + ", ".join([f"P{i}: {self.agents[i].name}" for i in range(self.num_players)]))
        self._write(r"")
        self._write(r"\vspace{0.2cm}")
        self._write(r"\hrule")
        self._write(r"\vspace{0.3cm}")
    
    def _write_turns_flow(self, turn_data_list: List):
        """Write turns in flow layout - side by side with wrapping."""
        page_width = 19.0  # cm available width
        turn_gap = 0.6  # gap between turn containers
        row_height = 4.0  # height of each row
        
        # Calculate widths for all turns
        turn_widths = [self._calculate_turn_width(td) for td in turn_data_list]
        
        # Layout algorithm: pack turns into rows
        rows = []
        current_row = []
        current_row_width = 0
        
        for i, (hand_w, move_w, total_w) in enumerate(turn_widths):
            if not current_row:
                # First turn in row
                current_row.append((i, hand_w, move_w, total_w))
                current_row_width = total_w
            elif current_row_width + turn_gap + total_w <= page_width:
                # Fits in current row
                current_row.append((i, hand_w, move_w, total_w))
                current_row_width += turn_gap + total_w
            else:
                # Start new row
                rows.append(current_row)
                current_row = [(i, hand_w, move_w, total_w)]
                current_row_width = total_w
        
        if current_row:
            rows.append(current_row)
        
        # Draw all turns in flow layout
        self._write(r"\begin{tikzpicture}[")
        self._write(r"  turn/.style={draw=black!70, rounded corners=3pt, line width=0.8pt, fill=white},")
        self._write(r"  inner/.style={draw=black!50, rounded corners=2pt, line width=0.6pt},")
        self._write(r"  arrow/.style={-{Stealth[length=2.5mm]}, line width=0.8pt, blue!60!black}")
        self._write(r"]")
        
        y_offset = 0
        turn_positions = {}  # Store (x, y) for each turn index
        
        for row_idx, row in enumerate(rows):
            x_offset = 0
            
            for pos_in_row, (turn_idx, hand_w, move_w, total_w) in enumerate(row):
                turn_data = turn_data_list[turn_idx]
                
                # Store position for arrow drawing
                turn_positions[turn_idx] = (x_offset + total_w / 2, y_offset - row_height / 2)
                
                # Draw turn container
                self._write(f"  % Turn {turn_idx + 1}")
                self._write(f"  \\node[turn, anchor=north west] at ({x_offset}, {y_offset}) {{")
                self._write_turn_content(turn_data, hand_w, move_w)
                self._write(r"  };")
                
                x_offset += total_w + turn_gap
            
            y_offset -= row_height + 0.5  # Move down for next row
        
        # Draw arrows between consecutive turns
        for i in range(len(turn_data_list) - 1):
            x1, y1 = turn_positions[i]
            x2, y2 = turn_positions[i + 1]
            
            if abs(y1 - y2) < 0.1:  # Same row - horizontal arrow
                self._write(f"  \\draw[arrow] ({x1 + 0.3}, {y1}) -- ({x2 - 0.3}, {y2});")
            else:  # Different rows - L-shaped arrow
                mid_y = (y1 + y2) / 2
                self._write(f"  \\draw[arrow] ({x1}, {y1 - 0.3}) -- ({x1}, {mid_y}) -- ({x2}, {mid_y}) -- ({x2}, {y2 + 0.3});")
        
        self._write(r"\end{tikzpicture}")
        self._write(r"")
    
    def _write_turn_content(self, turn_data, hand_w: float, move_w: float):
        """Write content of a turn container."""
        turn = turn_data['turn']
        state = turn_data['state']
        post_state = turn_data.get('post_state')
        agent = turn_data['agent']
        move = turn_data['move']
        legal_moves = turn_data['legal_moves']
        
        player = state.get_current_player()
        hand_size = player.hand_size()
        
        # Header
        header = f"Turn {turn}: P{state.current_player} ({agent.name})"
        if state.dice_state.wild_active:
            header += " [W]"
        if state.dice_state.double_play_active:
            header += " [D]"
        
        self._write(r"    \begin{tikzpicture}[baseline=0pt]")
        self._write(r"      \node[font=\scriptsize\bfseries, anchor=north west] at (0, 0) {" + header + r"};")
        
        # Calculate scaling
        self._write(r"      \pgfmathsetmacro{\Nc}{" + f"{max(hand_size, 1)}" + r"}")
        self._write(r"      \pgfmathsetmacro{\rawW}{\Nc * \CardWo + (\Nc - 1) * \CGapo}")
        self._write(r"      \pgfmathsetmacro{\cs}{min(1.0, \MaxRowW / \rawW)}")
        self._write(r"      \pgfmathsetmacro{\cw}{\CardWo * \cs}")
        self._write(r"      \pgfmathsetmacro{\ch}{\CardHo * \cs}")
        self._write(r"      \pgfmathsetmacro{\cg}{\CGapo * \cs}")
        
        # Container setup
        self._write(r"      \def\pad{0.20}")
        self._write(r"      \def\gap{0.25}")
        self._write(r"      \def\handW{" + f"{hand_w}" + r"}")
        self._write(r"      \def\moveW{" + f"{move_w}" + r"}")
        self._write(r"      \def\containerH{2.2}")
        self._write(r"      \def\startY{-0.35}")
        
        # Hand container
        self._write(r"      \draw[inner] (0, \startY) rectangle (\handW, \startY - \containerH);")
        self._write(r"      \node[font=\tiny, anchor=north west] at (\pad, \startY - 0.05) {Hand (" + f"{hand_size}" + r")};")
        
        # Draw cards
        self._write_hand_cards(player.hand, legal_moves)
        
        # Move container
        self._write(r"      \pgfmathsetmacro{\moveX}{\handW + \gap}")
        self._write(r"      \draw[inner] (\moveX, \startY) rectangle (\moveX + \moveW, \startY - \containerH);")
        self._write(r"      \node[font=\tiny, anchor=north] at (\moveX + \moveW/2, \startY - 0.05) {Move};")
        
        # Draw move
        self._write_move_content(move, state, post_state)
        
        self._write(r"    \end{tikzpicture}")
    
    def _write_hand_cards(self, hand: List[Card], legal_moves: List):
        """Draw hand cards."""
        if not hand:
            self._write(r"      \node[text=gray!60, font=\tiny] at (\handW/2, \startY - \containerH/2) {empty};")
            return
        
        playable_cards = {move.card for move in legal_moves if isinstance(move, PlayCard)}
        sorted_hand = sorted(hand, key=lambda c: (self.SUIT_TO_LATEX[c.suit], self.RANK_TO_INDEX[c.rank]))
        
        for i, card in enumerate(sorted_hand):
            suit_latex = self.SUIT_TO_LATEX[card.suit]
            rank_idx = self.RANK_TO_INDEX[card.rank]
            
            self._write(f"      \\pgfmathsetmacro{{\\xc}}{{\\pad + {i} * (\\cw + \\cg)}}")
            self._write(r"      \pgfmathsetmacro{\yc}{\startY - \containerH + \pad}")
            self._write(f"      \\DrawCardScaled{{\\xc}}{{\\yc}}{{{rank_idx}}}{{{suit_latex}}}{{\\cs}}")
            
            if card in playable_cards:
                self._write(r"      \draw[green!70!black, line width=1.2pt] (\xc, \yc) rectangle (\xc + \cw, \yc + \ch);")
    
    def _write_move_content(self, move, old_state: GameState, new_state: GameState = None):
        """Draw move content."""
        if isinstance(move, PlayCard):
            suit_latex = self.SUIT_TO_LATEX[move.card.suit]
            rank_idx = self.RANK_TO_INDEX[move.card.rank]
            
            self._write(r"      \pgfmathsetmacro{\cardX}{\moveX + (\moveW - \CardWo) / 2}")
            self._write(r"      \pgfmathsetmacro{\cardY}{\startY - \containerH + \pad}")
            self._write(f"      \\DrawCardScaled{{\\cardX}}{{\\cardY}}{{{rank_idx}}}{{{suit_latex}}}{{1.0}}")
            
        elif isinstance(move, RollDice):
            outcome = self._get_dice_outcome_text(old_state, new_state) if new_state else "..."
            self._write(r"      \node[font=\tiny, text width=\moveW cm, align=center, anchor=north] at (\moveX + \moveW/2, \startY - 0.5) {")
            self._write(r"        \textbf{RollDice}\\[0.08cm]")
            self._write(r"        " + outcome)
            self._write(r"      };")
            
        elif isinstance(move, Pass):
            pass_text = "forced" if not move.voluntary else f"vol. -{old_state.variant.voluntary_pass_penalty}pt"
            self._write(r"      \node[font=\tiny, text width=\moveW cm, align=center, anchor=north] at (\moveX + \moveW/2, \startY - 0.5) {")
            self._write(r"        \textbf{Pass}\\[0.08cm]")
            self._write(r"        (" + pass_text + r")")
            self._write(r"      };")
    
    def _get_dice_outcome_text(self, old_state: GameState, new_state: GameState) -> str:
        """Get dice outcome text for display."""
        old_player = old_state.get_current_player()
        new_player = new_state.players[old_state.current_player]
        
        if new_state.dice_state.wild_active and not old_state.dice_state.wild_active:
            return r"{\color{green!60!black}WILD}"
        
        if new_state.dice_state.double_play_active and not old_state.dice_state.double_play_active:
            return r"{\color{green!60!black}DOUBLE}"
        
        old_hand = old_player.hand_size()
        new_hand = new_player.hand_size()
        
        if new_hand < old_hand:
            for i, p in enumerate(new_state.players):
                if i != old_state.current_player and p.hand_size() > old_state.players[i].hand_size():
                    return f"{{\\color{{green!60!black}}GIVE to P{i}}}"
            return r"{\color{green!60!black}GIVE}"
        
        if new_hand > old_hand:
            for i, p in enumerate(new_state.players):
                if i != old_state.current_player and p.hand_size() < old_state.players[i].hand_size():
                    return f"{{\\color{{red!60!black}}TAKE from P{i}}}"
            return r"{\color{red!60!black}TAKE}"
        
        if new_state.dice_state.revealed_player is not None:
            if new_state.dice_state.revealed_player == old_state.current_player:
                return r"{\color{red!60!black}REVEALED}"
            return f"{{\\color{{green!60!black}}SEE P{new_state.dice_state.revealed_player}}}"
        
        return "[unknown]"
    
    def _write_final_state(self, turn_data):
        """Write final game state."""
        state = turn_data['state']
        
        self._write(r"\vspace{0.5cm}")
        self._write(r"\noindent\textbf{\Large Game Over}")
        self._write(r"")
        
        winner_name = self.agents[state.winner].name
        self._write(r"\noindent\textbf{Winner:} P" + f"{state.winner}" + r" (" + f"{winner_name}" + r")")
        self._write(r"")
        
        self._write(r"\vspace{0.3cm}")
        self._write(r"\begin{center}")
        self._write(r"\begin{tabular}{|c|l|c|c|}")
        self._write(r"\hline")
        self._write(r"\textbf{P} & \textbf{Agent} & \textbf{Round} & \textbf{Match} \\ \hline")
        
        for i, player in enumerate(state.players):
            mark = "$\\star$" if i == state.winner else ""
            self._write(f"{i} {mark} & {self.agents[i].name} & {player.round_score:+d} & {player.match_score:+d} \\\\ \\hline")
        
        self._write(r"\end{tabular}")
        self._write(r"\end{center}")


def main():
    """Generate LaTeX visualization."""
    from game.entities import ScoringMode, MatchEndMode
    from datetime import datetime
    
    agents = [
        MCTSAgentDeep(),
        RLAgent(),
        create_balanced_heuristic(),
        create_aggressive_heuristic()
    ]
    
    variant = VariantConfig(
        scoring_mode=ScoringMode.DOUBLE_PENALTY,
        match_end_mode=MatchEndMode.FIXED_ROUNDS,
        fixed_rounds_count=1,
        points_per_card=1,
        voluntary_pass_penalty=1
    )
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"game_flow_{timestamp}.tex"
    
    visualizer = LaTeXGameVisualizer(agents, variant, output_file=output_file)
    visualizer.play_and_visualize()


if __name__ == "__main__":
    main()