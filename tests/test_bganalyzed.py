"""Tests for the GNUBG analysis-export parser."""

from raccoon.data.bganalyzed import (
    parse_analyzed,
    best_candidates,
)


# A compact but faithful 3-decision snippet of a GNUBG 4-ply text export:
#  - move 1: played move == rank 1 (best); one 4-ply and one 2-ply candidate
#            (exercises deepest-ply filtering)
#  - move 2: a blunder — the played move is rank 2, not rank 1
#  - move 3: a forfeit ("cannot move")
SAMPLE = """The score (after 0 games) is: Magic 0, Lasse 0 (match to 5 points)

5 point match

Move number 1:  Magic to play 32

 GNU Backgammon  Position ID: 4HPwATDgc/ABMA
                 Match ID   : MICpAAAAAAAE
 +12-11-10--9--8--7-------6--5--4--3--2--1-+     O: Magic
 | X           O    |   | O              X |     0 points
 +13-14-15-16-17-18------19-20-21-22-23-24-+     X: Lasse
Pip counts: O 167, X 167

* Magic moves 24/21 13/11

Rolled 32:
*    1. Cubeful 4-ply    24/21 13/11                  Eq.: +0.002
       0.501 0.136 0.006 - 0.499 0.134 0.006
     2. Cubeful 2-ply    13/8                         Eq.: -0.039 (-0.041)
       0.490 0.136 0.006 - 0.510 0.142 0.006


Move number 2:  Lasse to play 41

 GNU Backgammon  Position ID: 4HPkASLgc/ABMA
                 Match ID   : cAmmAAAAAAAE
 +13-14-15-16-17-18------19-20-21-22-23-24-+     O: Magic
 | X  O        O    |   | O              X |     0 points
 +12-11-10--9--8--7-------6--5--4--3--2--1-+     X: Lasse
Pip counts: O 162, X 167

* Lasse moves 8/4*/3

Rolled 41:
     1. Cubeful 4-ply    24/23 8/4*                   Eq.: -0.168
       0.460 0.125 0.005 - 0.540 0.158 0.009
*    2. Cubeful 4-ply    8/4*/3                       Eq.: -0.214 (-0.045)
       0.450 0.121 0.005 - 0.550 0.164 0.010


Move number 3:  Magic to play 55

 GNU Backgammon  Position ID: 4HPkASLgc/ABZA
                 Match ID   : MIGmAAAAAAAE
 +12-11-10--9--8--7-------6--5--4--3--2--1-+     O: Magic
 | X           O    |   | O              X |     0 points
 +13-14-15-16-17-18------19-20-21-22-23-24-+     X: Lasse
Pip counts: O 130, X 163

* Magic cannot move

Rolled 55:
*    Cannot move


Game statistics for game 1
"""


def test_player_mapping_and_match_length():
    g = parse_analyzed(SAMPLE)
    assert g.player_x == "Lasse"   # X == OpenSpiel player 0
    assert g.player_o == "Magic"   # O == OpenSpiel player 1
    assert g.match_length == 5


def test_decision_count_and_fields():
    g = parse_analyzed(SAMPLE)
    assert len(g.decisions) == 3
    d1 = g.decisions[0]
    assert d1.move_number == 1
    assert d1.player_name == "Magic"
    assert d1.dice == (3, 2)
    assert d1.position_id == "4HPwATDgc/ABMA"
    assert d1.played_move_str == "24/21 13/11"
    assert len(d1.candidates) == 2


def test_deepest_ply_filter():
    g = parse_analyzed(SAMPLE)
    # move 1 has a 4-ply and a 2-ply candidate; keep only the 4-ply one.
    deep = best_candidates(g.decisions[0])
    assert len(deep) == 1
    assert deep[0].ply == 4
    assert deep[0].move_str == "24/21 13/11"
    assert deep[0].is_played is True


def test_probability_tuple_parsed():
    g = parse_analyzed(SAMPLE)
    c = best_candidates(g.decisions[0])[0]
    assert c.probs == (0.501, 0.136, 0.006, 0.499, 0.134, 0.006)


def test_blunder_played_not_rank1():
    g = parse_analyzed(SAMPLE)
    d2 = g.decisions[1]
    assert d2.played_move_str == "8/4*/3"
    # rank 1 is the best move, rank 2 is what was played
    by_rank = {c.rank: c for c in d2.candidates}
    assert by_rank[1].is_played is False
    assert by_rank[1].move_str == "24/23 8/4*"
    assert by_rank[2].is_played is True


def test_forfeit_cannot_move():
    g = parse_analyzed(SAMPLE)
    d3 = g.decisions[2]
    assert d3.dice == (5, 5)
    assert d3.played_move_str is None       # forfeit
    assert d3.candidates == []              # no checker candidates


def test_stats_section_terminates_parsing():
    g = parse_analyzed(SAMPLE)
    # "Game statistics" footer must not become a 4th decision.
    assert len(g.decisions) == 3
