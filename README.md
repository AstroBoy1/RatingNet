# Chess Rating Estimation from Moves and Clock Times Using a CNN-LSTM

**Authors**: Michael Omori (ORCID: [0009-0000-4632-9272](https://orcid.org/0009-0000-4632-9272)), Prasad Tadepalli (ORCID: [0000-0003-2736-3912](https://orcid.org/0000-0003-2736-3912))  
Oregon State University, Corvallis OR, USA  
Email: [omorim@oregonstate.edu](mailto:omorim@oregonstate.edu), [prasad.tadepalli@oregonstate.edu](mailto:prasad.tadepalli@oregonstate.edu)

---

## Abstract

Current rating systems update ratings incrementally and may not always accurately reflect a player's true strength at all times, especially for rapidly improving or rusty players. To overcome this, we explore a method to estimate player ratings directly from game moves and clock times. We compiled a benchmark dataset from Lichess, encompassing various time controls and including move sequences and clock times.

Our model architecture comprises a CNN to learn positional features, which are then integrated with clock-time data into a bidirectional LSTM, predicting player ratings after each move. The model achieved a Mean Absolute Error (MAE) of 182 rating points on the test data. Additionally, we applied our model to the 2024 IEEE Big Data Cup Chess Puzzle Difficulty Competition dataset, predicted puzzle ratings, and achieved competitive results.

This model is the first to use no hand-crafted features to estimate chess ratings and the first to output a rating prediction for each move. Our method highlights the potential of using move-based rating estimation for enhancing rating systems and possibly for applications such as cheating detection.
