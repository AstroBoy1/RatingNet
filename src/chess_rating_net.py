import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
import os
import pickle
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
import time


def time_to_seconds(time_str):
    """Converts a time string of the format 'HH:MM:SS' to seconds."""
    parts = time_str.split(':')
    return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])


class ChessGamesDataset(Dataset):
    def __init__(self, filenames, max_moves=100, ratings_mean=1514, ratings_std=366, clocks_mean=273, clocks_std=380):
        self.filenames = filenames
        self.max_moves = max_moves
        self.ratings_mean = ratings_mean
        self.ratings_std = ratings_std
        self.clocks_mean = clocks_mean
        self.clocks_std = clocks_std
        
    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        with open(self.filenames[idx], 'rb') as f:
            game_info = pickle.load(f)
        clocks = [time_to_seconds(c) for c in game_info.get('Clocks', [])]
        clocks = [(c - self.clocks_mean) / self.clocks_std for c in clocks]
        clocks = torch.tensor(clocks, dtype=torch.float)[:self.max_moves]
        # Ablation to set clocks to 0
        clocks = torch.zeros_like(clocks)
        white = False
        if "white" in game_info:
            white = game_info["white"]
        last_rating = None
        if "rating_after_last_game" in game_info:
            last_rating = game_info["rating_after_last_game"]
            last_rating = (last_rating - self.ratings_mean) / self.ratings_std
            last_rating = torch.tensor(last_rating, dtype=torch.float)
        positions = torch.stack(game_info['Positions'])[:self.max_moves]
        white_elo, black_elo = float(game_info['WhiteElo']), float(game_info['BlackElo'])
        targets = torch.tensor([white_elo, black_elo], dtype=torch.float)
        targets = (targets - self.ratings_mean) / self.ratings_std

        length = len(positions)
        initial_time, increment = map(int, game_info['Time'].split('+'))
        estimated_duration = initial_time + 40 * increment
        time_control = self.categorize_time_control(estimated_duration)

        result = None
        if "Result" in game_info:
            result = game_info["Result"]

        return {'positions': positions, 'clocks': clocks, 'targets': targets, 'length': length, 'time_control': time_control, 
        'white': white, 'last_rating': last_rating, 'result': result}
    
    def categorize_time_control(self, estimated_duration):
        if estimated_duration < 29:
            return 'ultrabullet'
        elif estimated_duration < 179:
            return 'bullet'
        elif estimated_duration < 479:
            return 'blitz'
        elif estimated_duration < 1499:
            return 'rapid'
        else:
            return 'classical'

def collate_fn(batch):
    positions = pad_sequence([item['positions'] for item in batch], batch_first=True)
    clocks = pad_sequence([item['clocks'] for item in batch], batch_first=True)
    targets = torch.stack([item['targets'] for item in batch])
    lengths = torch.tensor([item['length'] for item in batch], dtype=torch.int)
    time_controls = [item['time_control'] for item in batch]
    white = torch.tensor([item['white'] for item in batch])
    last_rating = None
    if batch[0]['last_rating']:
        last_rating = torch.stack([item['last_rating'] for item in batch])
    if batch[0]['result']:
        results = [item['result'] for item in batch]
        return {'positions': positions, 'clocks': clocks, 'targets': targets, 'lengths': lengths, 'time_controls': time_controls, 'white': white, 'last_rating': last_rating, 'results': results}
    return {'positions': positions, 'clocks': clocks, 'targets': targets, 'lengths': lengths, 'time_controls': time_controls, 'white': white, 'last_rating': last_rating}


class ChessEloPredictor(nn.Module):
    def __init__(self, conv_filters=16, lstm_layers=2, dropout_rate=0.5, lstm_h=64, fc1_h=16, bidirectional=False):
        super(ChessEloPredictor, self).__init__()
        self.conv1 = nn.Conv2d(12, conv_filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(conv_filters)
        self.conv2 = nn.Conv2d(conv_filters, conv_filters * 2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(conv_filters * 2)
        self.conv3 = nn.Conv2d(conv_filters*2, conv_filters * 4, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(conv_filters * 4)
        self.conv4 = nn.Conv2d(conv_filters*4, conv_filters * 8, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(conv_filters * 8)
        self.pool = nn.AvgPool2d(2, 2)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.lstm = nn.LSTM(input_size=conv_filters * 8 + 1, hidden_size=lstm_h, num_layers=lstm_layers, batch_first=True, bidirectional=bidirectional)
        self.fc1 = nn.Linear(lstm_h, fc1_h)
        if bidirectional:
            self.fc1 = nn.Linear(lstm_h * 2, fc1_h)
        self.fc2 = nn.Linear(fc1_h, 2)


    def forward(self, positions, clocks, lengths):
        batch_size = positions.size(0)
        sequence_length = positions.size(1)
        positions = positions.view(-1, 12, 8, 8)
        x = F.leaky_relu(self.bn1(self.conv1(positions)))
        x = self.pool(x)
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = F.leaky_relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = F.leaky_relu(self.bn4(self.conv4(x)))
        x = self.dropout1(x)
        x = x.view(batch_size, sequence_length, -1)
        # [batch=32, sequence_length, hidden=256]
        clocks = clocks.unsqueeze(2)
        # [batch, sequence_length, 1]
        lstm_input = torch.cat((x, clocks), dim=2)
        # [batch=32, sequence_length, hidden=258]
        packed_input = pack_padded_sequence(lstm_input, lengths, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed_input)
        lstm_output, _ = pad_packed_sequence(packed_output, batch_first=True)

        # Take the last time step
        #lstm_output = lstm_output[torch.arange(lstm_output.size(0)), lengths - 1]
        # [batch, hidden_dim=128]
        y = F.leaky_relu(self.fc1(lstm_output))
        y = self.dropout1(y)
        y = self.fc2(y)

        # Use torch.arange to select the last time step for each sequence
        idx = torch.arange(batch_size)
        last_time_step_output = y[idx, lengths - 1, :]
        return y, last_time_step_output
    

def train_one_epoch(model, train_loader, device, criterion, optimizer, ratings_mean=1514, ratings_std=366):
    model.train()
    total_train_loss = 0
    for batch in train_loader:
        positions = batch['positions'].to(device)
        clocks = batch['clocks'].to(device)
        targets = batch['targets'].to(device)
        lengths = batch['lengths']
        optimizer.zero_grad()
        all, outputs = model(positions, clocks, lengths)
        loss = criterion(outputs * ratings_std + ratings_mean, targets * ratings_std + ratings_mean)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
    return total_train_loss / len(train_loader)


def validate(model, val_loader, device, criterion, ratings_mean=1514, ratings_std=366):
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            positions = batch['positions'].to(device)
            clocks = batch['clocks'].to(device)
            targets = batch['targets'].to(device)
            lengths = batch['lengths']
            all, outputs = model(positions, clocks, lengths)
            loss = criterion(outputs * ratings_std + ratings_mean, targets * ratings_std + ratings_mean)
            total_val_loss += loss.item()
    return total_val_loss / len(val_loader)

def mae_per_item(outputs, targets, ratings_mean, ratings_std):
    outputs_rescaled = outputs * ratings_std + ratings_mean
    targets_rescaled = targets * ratings_std + ratings_mean
    mae = torch.abs(outputs_rescaled - targets_rescaled)
    return mae.mean(dim=1)  # Mean absolute error per item

def mse_per_item(outputs, targets, ratings_mean, ratings_std):
    outputs_rescaled = outputs * ratings_std + ratings_mean
    targets_rescaled = targets * ratings_std + ratings_mean
    mse = (outputs_rescaled - targets_rescaled) ** 2
    return mse.mean(dim=1)  # Mean squared error per item

def loss_results(ratings, results):
    """
    Function to compute the accuracy of rating-based predictions.
    +1 if white rating is higher than black rating and white won.
    +1 if black rating is higher than white rating and black won.
    
    :param ratings: List or tensor of [white_rating, black_rating] pairs.
    :param results: List or tensor of game results (1 for white win, 0 for draw, -1 for black win).
    :return: Total correct predictions.
    """
    correct_predictions = 0
    total_predictions = 0

    for (white_rating, black_rating), result in zip(ratings, results):
        if white_rating > black_rating and result == '1-0':  # White wins with higher rating
            correct_predictions += 1
            total_predictions += 1
        elif black_rating > white_rating and result == '0-1':  # Black wins with higher rating
            correct_predictions += 1
            total_predictions += 1
        elif white_rating > black_rating and result == '0-1':  # White loses with higher rating
            total_predictions += 1
        elif black_rating > white_rating and result == '1-0': # Black loses with higher rating
            total_predictions += 1
    return correct_predictions, total_predictions


def test(model, test_loader, device, criterion, ratings_mean=1514, ratings_std=366):
    model.eval()
    total_test_loss = 0
    total_correct_predictions = 0
    total_games = 0
    loss_by_time_control = {'ultrabullet': 0, 'bullet': 0, 'blitz': 0, 'rapid': 0, 'classical': 0}
    count_by_time_control = {'ultrabullet': 0, 'bullet': 0, 'blitz': 0, 'rapid': 0, 'classical': 0}

    with torch.no_grad():
        for batch in test_loader:
            positions = batch['positions'].to(device)
            clocks = batch['clocks'].to(device)
            targets = batch['targets'].to(device)
            lengths = batch['lengths']
            time_controls = batch['time_controls']
            game_results = batch['results']  # Assuming results are part of the batch

            #correct_predictions, total_predictions = loss_results(targets, game_results)
            #correct_predictions, total_predictions = loss_results(outputs, game_results)
            #total_correct_predictions += correct_predictions
            #total_games += total_predictions
            #continue

            all, outputs = model(positions, clocks, lengths)
            # Test setting outputs to mean rating
            outputs = torch.zeros_like(outputs)
            loss = criterion(outputs * ratings_std + ratings_mean, targets * ratings_std + ratings_mean)

            total_test_loss += loss.item()

            #mae = mae_per_item(outputs, targets, ratings_mean, ratings_std)
            mse = mse_per_item(outputs, targets, ratings_mean, ratings_std)

            for idx, time_control in enumerate(time_controls):
                #loss_by_time_control[time_control] += mae[idx].item()
                loss_by_time_control[time_control] += mse[idx].item()
                count_by_time_control[time_control] += 1

            # Rating accuracy calculation
            #correct_predictions, total_predictions = loss_results(targets, game_results)
            #correct_predictions, total_predictions = loss_results(outputs, game_results)
            #total_correct_predictions += correct_predictions
            #total_games += total_predictions
    #return total_correct_predictions / total_games
    for key in loss_by_time_control:
        if count_by_time_control[key] > 0:
            loss_by_time_control[key] /= count_by_time_control[key]

    #accuracy = total_correct_predictions / total_games if total_games > 0 else 0
    return total_test_loss / len(test_loader), loss_by_time_control


def test_new(model, test_loader, device, criterion, ratings_mean=1514, ratings_std=366):
    """
    Tests the model on new players and compares predicted rating after the first game 
    with their last known rating (gold standard).
    """
    model.eval()
    total_test_loss = 0
    total_targets_loss = 0

    with torch.no_grad():
        for batch in test_loader:
            positions = batch['positions'].to(device)
            clocks = batch['clocks'].to(device)
            targets = batch['targets'].to(device)  # Actual Elo ratings (normalized)
            lengths = batch['lengths']
            white_flags = batch['white'].to(device)  # True if the new player is White
            last_ratings = batch['last_rating'].to(device)  # Last Elo ratings (gold standard)

            # Forward pass through the model
            all, outputs = model(positions, clocks, lengths)
            #breakpoint()
            # Rescale the model outputs and targets to actual Elo ratings
            outputs_rescaled = outputs * ratings_std + ratings_mean
            targets_rescaled = last_ratings * ratings_std + ratings_mean

            # Select either White or Black player's prediction based on the `white` flag
            predicted_ratings = torch.where(white_flags, outputs_rescaled[:, 0], outputs_rescaled[:, 1])
            new_targets = torch.where(white_flags, targets[:, 0], targets[:, 1])

            # Compute the loss (difference between predicted and last known rating)
            loss = criterion(predicted_ratings, targets_rescaled)
            total_test_loss += loss.item()
            new_targets = new_targets * ratings_std + ratings_mean
            targets_loss = criterion(new_targets, targets_rescaled)
            total_targets_loss += targets_loss.item()
            breakpoint()
    return total_test_loss / len(test_loader), total_targets_loss / len(test_loader)

def main():
    data_dir = "data/processed_games"
    experiment_name = "cnn_bilstm_clocks_all"

    load_model = False
    
    # params = {
    #     'train_batch_size': 32,
    #     'val_batch_size': 8192,
    #     'num_workers': 4,
    #     'learning_rate': 0.0001,
    #     'weight_decay': 1e-5,
    #     'epochs': 100,
    #     'optimizer': 'Adam',
    #     'patience': 5,
    #     'lr_factor': 0.5,
    #     "conv_filters":32,
    #     "lstm_layers":3,
    #     "bidirectional":True,
    #     "dropout_rate":0.5,
    #     "lstm_h":64,
    #     "fc1_h":32
    # }
    params = {
        'train_batch_size': 32,
        'val_batch_size': 8192,
        'num_workers': 16,
        'learning_rate': 0.0001,
        'weight_decay': 1e-5,
        'epochs': 60,
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

    all_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.pkl')]
    model_dir = os.path.join('models', experiment_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    log_dir = os.path.join('runs', experiment_name)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    writer = SummaryWriter(log_dir=log_dir)

    # Split into train, val, and test sets
    train_val_files, test_files = train_test_split(all_files, test_size=0.1, random_state=42)
    train_files, val_files = train_test_split(train_val_files, test_size=0.2, random_state=42)

    train_dataset = ChessGamesDataset(train_files)
    val_dataset = ChessGamesDataset(val_files)
    test_dataset = ChessGamesDataset(test_files)

    train_loader = DataLoader(train_dataset, batch_size=params["train_batch_size"], shuffle=True, collate_fn=collate_fn, num_workers=params['num_workers'])
    val_loader = DataLoader(val_dataset, batch_size=params['val_batch_size'], shuffle=False, collate_fn=collate_fn, num_workers=params['num_workers'])
    test_loader = DataLoader(test_dataset, batch_size=params['val_batch_size'], shuffle=False, collate_fn=collate_fn, num_workers=params['num_workers'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)
    model = ChessEloPredictor(params["conv_filters"], params["lstm_layers"], params["dropout_rate"],
                              params["lstm_h"], params["fc1_h"], params["bidirectional"]).to(device)
    if load_model:
        saved_model = torch.load("models/cnn_bilstm_21-24/model_56.pth")
        params = saved_model["params"]
        model.load_state_dict(saved_model["model_state_dict"])

        criterion1 = nn.L1Loss()
        #criterion2 = nn.MSELoss()
        print("Testing Model")
        #train_loss = validate(model, train_loader, device, criterion1)
        #val_loss = validate(model, val_loader, device, criterion1)
        test_loss = test(model, test_loader, device, criterion1)
        #print(f'Train Loss: {train_loss:.4f}')
        #print(f'Validation Loss: {val_loss:.4f}')
        print("Test Loss: ", test_loss)
        return 1
        #print("MSE Train Loss: ", validate(model, train_loader, device, criterion2))
        #print("MSE Val Loss: ", validate(model, val_loader, device, criterion2))
        #print("MSE Test Loss: ", validate(model, test_loader, device, criterion2))

        # 10 random inference predictions
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                positions = batch['positions'].to(device)
                clocks = batch['clocks'].to(device)
                targets = batch['targets'].to(device)
                lengths = batch['lengths']
                outputs = model(positions, clocks, lengths)
                print("outputs", outputs * 366 + 1514)
                print("targets", targets * 366 + 1514)
                print("diff", ((outputs * 366 + 1514) - (targets * 366 + 1514)))
                diff = torch.abs((outputs*366+1514) - (targets*366+1514))
                mae = torch.mean(diff, dim=0)
                print("mae", mae.tolist())
                return 1
    
    optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=params['patience'], factor=params['lr_factor'])
    criterion = nn.L1Loss()
    best_val_loss = float('inf')
    best_epoch = 0
    best_path = None
    print("Training model")
    start = time.time()

    for epoch in range(params['epochs']):
        epoch_start = time.time()
        train_loss = train_one_epoch(model, train_loader, device, criterion, optimizer)
        print(f'Epoch {epoch + 1}, Train Loss: {train_loss:.4f}')
        val_loss = validate(model, val_loader, device, nn.L1Loss())
        print(f'Epoch {epoch + 1}, Validation Loss: {val_loss:.4f}')
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Validation', val_loss, epoch)
        epoch_duration = (time.time() - epoch_start) / 60
        writer.add_scalar('Timing/Epoch Duration', epoch_duration, epoch)
        scheduler.step(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_path = os.path.join(model_dir, f'model_{epoch+1}.pth')
            torch.save({'model_state_dict': model.state_dict(),
                       'params': params},
                       best_path)
            print("Saved best model")
    end = time.time()
    print("Training duration: ", (end - start) / 60)
    print("best val loss: ", best_val_loss)
    print("best val epoch: ", best_epoch)
    writer.close()

    saved_model = torch.load(best_path)
    model.load_state_dict(saved_model["model_state_dict"])
    test_loss = test(model, test_loader, device, nn.L1Loss())
    print("Test Loss: ", test_loss)
    return 1

def main_new():
    data_dir = "data/selected_first_games"
    params = {
        'train_batch_size': 32,
        'val_batch_size': 8192,
        'num_workers': 0,
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

    all_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.pkl')]

    test_dataset = ChessGamesDataset(all_files)

    test_loader = DataLoader(test_dataset, batch_size=params['val_batch_size'], shuffle=False, collate_fn=collate_fn, num_workers=params['num_workers'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)
    model = ChessEloPredictor(params["conv_filters"], params["lstm_layers"], params["dropout_rate"],
                              params["lstm_h"], params["fc1_h"], params["bidirectional"]).to(device)
    saved_model = torch.load("models/cnn_bilstm_clocks_all/model_55.pth")
    params = saved_model["params"]
    model.load_state_dict(saved_model["model_state_dict"])

    criterion1 = nn.L1Loss()
    test_loss = test_new(model, test_loader, device, nn.L1Loss())
    print("Test Loss: ", test_loss)

if __name__ == "__main__":
    #main()
    data_dir = "data/processed_games"
    #main_new()
    # data_dir = "data/"
    params = {
        'train_batch_size': 8192,
        'val_batch_size': 8192,
        'num_workers': 4,
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

    all_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.pkl')]

    # Split into train, val, and test sets
    train_val_files, test_files = train_test_split(all_files, test_size=0.1, random_state=42)
    train_files, val_files = train_test_split(train_val_files, test_size=0.2, random_state=42)

    test_dataset = ChessGamesDataset(test_files)

    train_dataset = ChessGamesDataset(train_files)
    train_loader = DataLoader(train_dataset, batch_size=params["train_batch_size"], shuffle=True, collate_fn=collate_fn, num_workers=params['num_workers'])
    test_loader = DataLoader(test_dataset, batch_size=params['val_batch_size'], shuffle=False, collate_fn=collate_fn, num_workers=params['num_workers'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)
    model = ChessEloPredictor(params["conv_filters"], params["lstm_layers"], params["dropout_rate"],
                              params["lstm_h"], params["fc1_h"], params["bidirectional"]).to(device)
    saved_model = torch.load("models/cnn_bilstm_clocks_all/model_55.pth")
    params = saved_model["params"]
    model.load_state_dict(saved_model["model_state_dict"])

    criterion1 = nn.L1Loss()
    criterion2 = nn.MSELoss()
    #test_loss = test(model, test_loader, device, nn.L1Loss())
    test_loss = test(model, test_loader, device, criterion2)
    print("Test Loss: ", test_loss)
