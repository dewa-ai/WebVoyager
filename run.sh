#!/bin/bash
# Replace the API key below with your actual OpenAI API key that starts with sk-
nohup python -u run.py \
    --test_file ./data/test.jsonl \
    --api_key "sk-..." \
    --headless \
    --max_iter 20 \
    --min_iter 5 \
    --max_attached_imgs 5 \
    --temperature 1 \
    --fix_box_color \
    --page_load_timeout 60 \
    --implicit_wait 20 \
    --window_width 1920 \
    --window_height 1080 \
    --seed 42 > test_tasks.log &
