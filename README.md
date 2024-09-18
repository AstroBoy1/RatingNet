# Chess Rating Estimation from Moves and Clock Times Using a CNN-LSTM

**Authors**: Michael Omori (ORCID: [0009-0000-4632-9272](https://orcid.org/0009-0000-4632-9272)), Prasad Tadepalli (ORCID: [0000-0003-2736-3912](https://orcid.org/0000-0003-2736-3912))  
Oregon State University, Corvallis OR, USA  
Email: [omorim@oregonstate.edu](mailto:omorim@oregonstate.edu), [prasad.tadepalli@oregonstate.edu](mailto:prasad.tadepalli@oregonstate.edu)

---

## Abstract

Current rating systems update ratings incrementally and may not always accurately reflect a player's true strength at all times, especially for rapidly improving or rusty players. To overcome this, we explore a method to estimate player ratings directly from game moves and clock times. We compiled a benchmark dataset from Lichess, encompassing various time controls and including move sequences and clock times.

Our model architecture comprises a CNN to learn positional features, which are then integrated with clock-time data into a bidirectional LSTM, predicting player ratings after each move. The model achieved a Mean Absolute Error (MAE) of 182 rating points on the test data. Additionally, we applied our model to the 2024 IEEE Big Data Cup Chess Puzzle Difficulty Competition dataset, predicted puzzle ratings, and achieved competitive results.

This model is the first to use no hand-crafted features to estimate chess ratings and the first to output a rating prediction for each move. Our method highlights the potential of using move-based rating estimation for enhancing rating systems and possibly for applications such as cheating detection.

## Installation and Setup
```bash
conda create --name rating_env python=3.8
conda activate rating_env
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install scikit-learn==1.3.2
conda install tensorboard
pip install chess==1.10.0
```

You can download games from https://database.lichess.org/ in the .pgn.zst format.
Put them in data/game_zips.
Next run
```bash
sh format.sh
```
You can change the year and months in that file. This will run src/format_data.py which converts the games into a format suitable for the cnn input.
The converted game data will be saved in data/processed_games.
Download model_55.pth and put it in models/cnn_bilstm_clocks_all.
We also provide a direct download from google drive at this link: 
You can run the code with
```bash
python src/chess_rating_net.py
```
python src/game_analysis.py will output analyzed games with the rating predictions.

## License
This project is licensed under the MIT license - see LICENSE.