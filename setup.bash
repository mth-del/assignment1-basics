#!/bin/bash

# v2 test 
# for i in {1..3}; do
#   python cs336_basics/test_trainer.py  bpe_v2_time test_tiny_story_v1_${i} -d data/TinyStoriesV2-GPT4-train.txt -v 10000 > ts_v1_${i}.log
# done


# v3
for i in {1..3}; do
  python cs336_basics/test_trainer.py bpe_v3_time out_dir -d data/TinyStoriesV2-GPT4-train.txt -v 10000 > ts_v1_${i}.log -c 16 -m 8  --do_monitor 
done