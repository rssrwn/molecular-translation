import argparse

DEFAULT_VOCAB_PATH = "vocab.txt"
DEFAULT_MAX_SEQ_LEN = 256


def collate(batch):
    img_ids, imgs = tuple(zip(*batch))
    imgs_batch = torch.stack(imgs)


def main(args):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", type=str)
    parser.add_argument("--vocab_path", type=str, default=DEFAULT_VOCAB_PATH)
    parser.add_argument("--max_seq_len", type=int, default=DEFAULT_MAX_SEQ_LEN)

    parsed_args = parser.parse_args()
    main(parsed_args)
