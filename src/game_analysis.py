import torch
import pickle
import os
import numpy as np
from sklearn.model_selection import train_test_split
from chess_rating_net import ChessEloPredictor
from chess_rating_net import time_to_seconds

# This file will output some sample game predictions to text files

def calculate_mae(predictions, white_elo, black_elo):
    mae_white = np.abs(predictions[:, 0] - white_elo)
    mae_black = np.abs(predictions[:, 1] - black_elo)
    return np.mean(mae_white + mae_black)

def save_game_to_txt(file_path, lines, output_dir, suffix, mae):
    output_file_name = os.path.splitext(os.path.basename(file_path))[0] + f'_{suffix}.txt'
    output_file_path = os.path.join(output_dir, output_file_name)
    
    with open(output_file_path, 'w') as output_file:
        for line in lines:
            output_file.write(line + '\n')
        output_file.write(f'MAE: {mae}')

def load_model_and_sample_games_to_txt(model_path, data_dir, sample_size=5, output_dir='games_with_lowest_highest_mae'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    params = {
        'train_batch_size': 32,
        'val_batch_size': 8192,
        'num_workers': 16,
        'learning_rate': 0.0001,
        'weight_decay': 1e-5,
        'epochs': 100,
        'optimizer': 'Adam',
        'patience': 5,
        'lr_factor': 0.5,
        "conv_filters":32,
        "lstm_layers":3,
        "bidirectional":True,
        "dropout_rate":0.5,
        "lstm_h":64,
        "fc1_h":32
    }
    model = ChessEloPredictor(params["conv_filters"], params["lstm_layers"], params["dropout_rate"],
                              params["lstm_h"], params["fc1_h"], params["bidirectional"]).to(device)
    
    saved_model = torch.load(model_path)
    model.load_state_dict(saved_model["model_state_dict"])
    model.eval()
    
    all_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.pkl')]
    train_val_files, test_files = train_test_split(all_files, test_size=0.1, random_state=42)
    # Randomly select a few games from the test set
    sampled_files = np.random.choice(test_files, sample_size, replace=False)
    
    lowest_mae = float('inf')
    highest_mae = float('-inf')
    lowest_mae_lines = None
    highest_mae_lines = None
    lowest_mae_file = None
    highest_mae_file = None
    
    for file_path in sampled_files:
        with open(file_path, 'rb') as f:
            game_info = pickle.load(f)
        
        clocks = [time_to_seconds(c) for c in game_info.get('Clocks', [])]
        clocks = torch.tensor(clocks, dtype=torch.float).to(device)
        
        positions = torch.stack(game_info['Positions']).to(device)
        
        with torch.no_grad():
            outputs, last = model(positions.unsqueeze(0), clocks.unsqueeze(0), torch.tensor([len(positions)]))
            predictions = outputs.squeeze().cpu().numpy()
        
        ratings_mean = 1514
        ratings_std = 366
        predictions = predictions * ratings_std + ratings_mean
        white_elo, black_elo = float(game_info['WhiteElo']), float(game_info['BlackElo'])

        mae = calculate_mae(predictions, white_elo, black_elo)
        
        # Prepare data to save to a text file
        lines = []
        moves = game_info.get('Moves', [])
        clocks = clocks.cpu().numpy()
        for i in range(len(moves[:100])):
            white_pred = predictions[i, 0]  # White's predicted rating
            black_pred = predictions[i, 1]  # Black's predicted rating
            line = (f"Move: {moves[i]}, ClockTime: {clocks[i]}, "
                    f"PredictedWhiteRating: {white_pred}, PredictedBlackRating: {black_pred}, "
                    f"WhiteElo: {white_elo}, BlackElo: {black_elo}")
            lines.append(line)
        
        # Track the games with the lowest and highest MAE
        if mae < lowest_mae:
            lowest_mae = mae
            lowest_mae_lines = lines
            lowest_mae_file = file_path
        if mae > highest_mae:
            highest_mae = mae
            highest_mae_lines = lines
            highest_mae_file = file_path
    
    # Save the game with the lowest MAE
    if lowest_mae_lines is not None:
        save_game_to_txt(lowest_mae_file, lowest_mae_lines, output_dir, 'lowest_mae', lowest_mae)
    
    # Save the game with the highest MAE
    if highest_mae_lines is not None:
        save_game_to_txt(highest_mae_file, highest_mae_lines, output_dir, 'highest_mae', highest_mae)
    
    print(f'Games with the lowest and highest MAE saved to text files in {output_dir}')

# Example usage
model_path = 'models/cnn_bilstm_clocks_all/model_55.pth'
data_dir = 'data/processed_games'
sample_size = 120000
sample_size = 5
load_model_and_sample_games_to_txt(model_path, data_dir, sample_size=sample_size)
