# Project Goal

Build a backgammon AI that clearly outplays GNUBG at money game.

## Assumptions

Research and challenge these if needed.

- GNUBG uses TD reinforcement learning. A superior system will likely need a more modern framework (AlphaZero-style policy+value + MCTS).
- GNUBG outputs cubeless outcome probability distributions (win/lose normal, gammon, backgammon) then applies Janowski formulas for cube actions. A superior system may need a more sophisticated model, perhaps outputting cubeful equities directly.

## Requirements

- Trained using AlphaZero-style self-play with a ResNet policy-value network and MCTS
- Uses OpenSpiel for backgammon game logic
- Terminal interface for human play, using the same standard output as GNUBG
- Automated play against GNUBG CLI for benchmarking
- Game logging in standard backgammon notation
- Training metadata logging: network version, architecture, performance metrics
- Text-based communication protocol (RGP) inspired by UCI, for future frontend integration

## Success Criteria

Raccoon has a possitive average win in money games against GNUBG at "world class" settings over 1000+ games, with 95% confidence interval above and not including zero.
