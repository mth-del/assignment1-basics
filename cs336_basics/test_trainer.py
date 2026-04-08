from secrets import token_bytes
import time
import argparse
import pickle
import os


def save_model(merges, vocabulary, out_dir):
    serializable_vocab = {}
    for token_id, token_bytes in vocabulary.items():
        serializable_vocab[str(token_id)] = list(token_bytes)
    
    serializable_merges = []

    for (byte1, byte2) in merges:
        serializable_merges.append([byte1, byte2])
    
    with open(out_dir + "/vocab.pkl", 'wb') as f :
        pickle.dump(serializable_vocab, f)
    
    with open(out_dir + "/merges.kl", 'wb') as f :
        pickle.dump(serializable_merges, f)


def main():
    parser = argparse.ArgumentParser(
        description= "bpe train on openwebtext"
    )

    parser.add_argument(
        "trainer",
        type=str,
        choices=[
            'bpe_v1_time',
            'bpe_v2_time',
        ],
    )

    parser.add_argument(
        "out_dir",
        type=str,
    )

    parser.add_argument(
        "--vocab_size",
        "-v",
        type=int,
        default=32000,
    )

    parser.add_argument(
        "--data_path",
        "-d",
        type=str,
        default="./data/owt_train.txt",
    )

    args, unknown_args = parser.parse_known_args()
    print(f"{args=}")
    print(f"{unknown_args=}")
    vocab_size = args.vocab_size
    data_path = args.data_path
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    match args.trainer:
        case "bpe_v1_time":
            from cs336_basics.bpe_v1_time import BPE_Trainer
        case "bpe_v2_time":
            from cs336_basics.bpe_v2_time import BPE_Trainer
    
    bpe_trainer = BPE_Trainer()

    start_time = time.perf_counter()

    vocabulary, merges = bpe_trainer.train(data_path, vocab_size,
                                            ["<|endoftext|>"],
                                            *unknown_args)
    end_time = time.perf_counter()
    print(f"total train time : {end_time - start_time:.2f} seconds")
    save_model(merges, vocabulary, out_dir)

if __name__ == '__main__':
    main()