import re
import pickle
import torch
import chess
import chess.pgn
import time
import sys


# This file formats the data for deep learning CNN


def board_to_array(board):
    """Converts a chess board into a 12-layer 8x8 tensor representing the piece positions."""
    # Board('rnbqkbnr/pppppppp/8/8/8/3P4/PPP1PPPP/RNBQKBNR b KQkq - 0 1')
    board_array = torch.zeros((12, 8, 8), dtype=torch.float32)
    piece_map = board.piece_map()
    # {63: Piece.from_symbol('r'), 62: Piece.from_symbol('n')
    for square, piece in piece_map.items():
        # 1 pawn, 2 knight, 3 bishop, 4 rook, 5 queen, 6 king
        # Piece color is true if white
        # White pieces 0-5, black 6-11
        index = piece.piece_type - 1 + (6 if piece.color == chess.BLACK else 0)
        row, col = divmod(square, 8)
        board_array[index, 7 - row, col] = 1
    return board_array


def parse_game(game):
    """Parses game object to extract evaluations, clocks and positions."""
    time_control = game.headers.get('TimeControl', '')

    # Only take 5 0 blitz games
    # if time_control != "300+0":
    #     return None
        
    board = game.board()
    moves = []
    evaluations = []
    clocks = []
    #positions = [board_to_array(board)]
    positions = []
    node = game
    # Take in the mainline, annotated pgn might have multiple variations
    while node.variations:
        next_node = node.variation(0)
        move = next_node.move
        board.push(move)
        positions.append(board_to_array(board))
        moves.append(move.uci())
        comment = next_node.comment
        eval_match = re.search(r"\[%eval\s+([^\]]+)\]", comment)
        if eval_match:
            eval = eval_match.group(1)
            if '#' in eval:
                mate_moves = int(eval[1:]) if eval[1] != '-' else int(eval[2:])
                eval = str(-200 + (mate_moves - 1)) if eval.startswith('#-') else str(200 - (mate_moves - 1))
            evaluations.append(eval)
        
        clock_match = re.search(r"\[%clk\s+([^\]]+)\]", comment)
        if clock_match:
            clocks.append(clock_match.group(1))
        
        node = next_node

    if not evaluations or not clocks:
        return None  # Skip games without evaluations or clock times (correspondence chess)

    # Some games did have evals after 150
    if len(clocks) > len(evaluations) + 1:
        # Evals stop at move 150
        evaluations = evaluations[:150]
        clocks = clocks[:150]
        positions = positions[:150]
    # final move results in the end of the game
    elif len(evaluations) + 1 == len(clocks):
        result = game.headers.get("Result")
        if '1-0' == result:
            evaluations.append('300')  # Arbitrary high positive value for white win
        elif '0-1' == result:
            evaluations.append('-300')  # Arbitrary high negative value for black win
        else:
            evaluations.append('0')  # Draw



    if len(evaluations) != len(clocks):
        print(len(evaluations), len(clocks), len(positions))
        print(game_to_pgn_string(game))
        #breakpoint()
        return -1
    if len(evaluations) != len(positions):
        print(len(evaluations), len(clocks), len(positions))
        print(game_to_pgn_string(game))
        #breakpoint()
        return -1
    if len(clocks) != len(positions):
        print(len(evaluations), len(clocks), len(positions))
        print(game_to_pgn_string(game))
        #breakpoint()
        return -1
    return {
        "WhiteElo": game.headers.get("WhiteElo", None),
        "BlackElo": game.headers.get("BlackElo", None),
        "Result": game.headers.get("Result", None),
        "Evaluations": evaluations,
        "Clocks": clocks,
        "Positions": positions,
        "Time": time_control,
        "Moves": moves
    }


def process_pgn_file(filename):
    """Processes each game in a PGN file."""
    with open(filename) as pgn:
        while True:
            game = chess.pgn.read_game(pgn)
            if game is None:
                break
            game_info = parse_game(game)
            if game_info:
                yield game_info


def process_pgn_stream():
    """Processes each game from the PGN data read from stdin."""
    while True:
        game = chess.pgn.read_game(sys.stdin)
        if game is None:
            break
        game_info = parse_game(game)
        if game_info:
            yield game_info


def game_to_pgn_string(game):
    exporter = chess.pgn.StringExporter(headers=True, variations=True, comments=True)
    pgn_string = game.accept(exporter)
    return pgn_string


def main():
    #year = "2021"
    #month = "04"

    year = sys.argv[1]
    month = sys.argv[2]

    file_path = f"data/game_zips/lichess_db_standard_rated_{year}-{month}.pgn.zst"
    #"data/game_zips/lichess_db_standard_rated_2021-04.pgn.zst"
    # zstdcat data/game_zips/lichess_db_standard_rated_2021-04.pgn.zst | python src/format_data.py

    max_game_per_month = 30000
    out_dir = "data/processed_games/"
    game_count = 0
    start = time.time()
    for game_info in process_pgn_stream():
    #for game_info in process_pgn_file(file_path):
        if game_count % 1000 == 0:
            print(game_count)
        if game_info == -1:
            print("error", game_count)
            continue
            #breakpoint()
        filename = f"{out_dir}lichess_db_standard_rated_{year}-{month}_{game_count}.pkl"
        with open(filename, 'wb') as file:
           pickle.dump(game_info, file)
        game_count += 1
        if game_count >= max_game_per_month:
            print("Finished saving data")
            break
    end = time.time()
    print(file_path)
    print("seconds: ", round(end - start))


if __name__ == "__main__":
    main()
