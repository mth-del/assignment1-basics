#!/bin/bash

for i in {1..3}; do
  python cs336_basics/test_trainer.py  bpe_v2_time test_tiny_story_v1_${i} -d data/TinyStoriesV2-GPT4-train.txt -v 10000 > ts_v1_${i}.log
done