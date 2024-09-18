#!/bin/bash

year="2024"

# From month 8 to 8 inclusive
for month in $(seq 8 8); do
    # Format the month with a leading zero
    formatted_month=$(printf "%02d" $month)

    file_path="data/game_zips/lichess_db_standard_rated_${year}-${formatted_month}.pgn.zst"
    echo "Processing file: $file_path"
    zstdcat $file_path | python src/format_data.py $year $month
    if [ $? -eq 0 ]; then
        echo "Successfully processed $file_path"
    else
        echo "Failed to process $file_path" >&2
    fi
done
