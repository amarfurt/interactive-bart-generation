""" Script to interactively run BART. """

import glob
import os
import torch
import torch.nn.functional as F
from pytorch_lightning import seed_everything
from pytorch_lightning.core import LightningModule
from pytorch_lightning.core.saving import load_hparams_from_yaml
from transformers import BartTokenizer, BartForConditionalGeneration, AddedToken

from data_schema import SchemaFactory
from dataloader import SummarizationDataModule, BartBatch

BART_WHITESPACE_CHAR = u'\u0120'


class BartSummarizer(LightningModule):

    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args)

        self.annotation_schema = SchemaFactory.get_schema(args.dataset)
        special_tokens = self.annotation_schema.get_special_text_tokens()
        special_tokens = [AddedToken(t) for t in special_tokens]
        self.tokenizer = BartTokenizer.from_pretrained(
            args.model_name_or_path, additional_special_tokens=special_tokens,
        )
        self.model = BartForConditionalGeneration.from_pretrained(args.model_name_or_path)
        self.model.resize_token_embeddings(len(self.tokenizer))  # extend embedding matrices for special tokens

    def forward(self, batch: BartBatch):
        labels = batch.tgt.clone().detach()
        labels = labels[:, 1:].contiguous()  # remove decoder start token (EOS token for BART)
        labels[labels == self.tokenizer.pad_token_id] = -100  # set padded tokens to -100 to ignore in LM loss
        decoder_outputs = self.model(
            input_ids=batch.src,
            attention_mask=batch.mask_src,
            decoder_input_ids=batch.tgt[:, :-1],
            decoder_attention_mask=batch.mask_tgt[:, :-1],
            labels=labels,
            return_dict=True,
        )
        return decoder_outputs

    def decode(self, output_ids):
        """ Decode BPE tokens into text. """
        text = self.tokenizer.decode(output_ids, spaces_between_special_tokens=False)
        stst_start = self.annotation_schema.mapping['standardized sentence']['text_start']
        stst_end = self.annotation_schema.mapping['standardized sentence']['text_end']
        text = text.replace(f'{stst_end} {stst_start}', f'{stst_end}<sent>{stst_start}')
        text = text.replace(self.tokenizer.bos_token, '')
        text = text.replace(self.tokenizer.eos_token, '')
        text = ' '.join(text.split())
        return text

    def generate_with_prefix(self, batch, prefix=None):
        args = {
            'input_ids': batch.src,
            'attention_mask': batch.mask_src,
            'max_length': self.hparams.max_length,
            'min_length': self.hparams.min_length,
            'no_repeat_ngram_size': self.hparams.ngram_blocking,
            'length_penalty': self.hparams.length_penalty,
            'num_beams': 5,
            'use_cache': True,
            'early_stopping': True,
        }
        if prefix:
            args['decoder_input_ids'] = self.tokenizer.encode(prefix, return_tensors='pt')[:, :-1]
        beam_output = self.model.generate(**args)
        source = self.decode(batch.src[0].tolist())
        reference = self.decode(batch.tgt[0].tolist())
        candidate = self.decode(beam_output[0].tolist())
        return source, reference, candidate


def load_model(args):
    """ Loads BART from the single checkpoint in the model dir. """
    checkpoints = glob.glob(os.path.join(args.model_dir, '*.ckpt'))
    assert len(checkpoints) == 1
    checkpoint_path = checkpoints[0]

    # restore args from hparams file
    hparams = load_hparams_from_yaml(os.path.join(args.model_dir, 'version_0', 'hparams.yaml'))
    for key, value in hparams.items():
        if not hasattr(args, key):
            setattr(args, key, value)

    # restore model
    model = BartSummarizer.load_from_checkpoint(args=args, checkpoint_path=checkpoint_path)
    model.eval()
    return model


def get_refdoc(text_dir):
    """ Returns the refdoc's ID and dataset split, if found. """
    refdoc_id = None
    while not refdoc_id:
        refdoc_path = input('Give the path to the refdoc you would like to use as input: ')
        if os.path.exists(refdoc_path):
            refdoc_id = os.path.basename(refdoc_path).split('.')[0]
        else:
            print(f'Unknown refdoc path: {refdoc_path}')

    # find dataset split
    for split in ['train', 'valid', 'test']:
        assert os.path.exists(os.path.join(text_dir, f'{split}.txt')), f"Did not find file with {split} ids"
        with open(os.path.join(text_dir, f'{split}.txt'), 'r') as f:
            split_ids = set(map(str.strip, f.readlines()))
        if refdoc_id in split_ids:
            return refdoc_id, split
    print('Refdoc is not in list of known ids. Cannot handle unknown files for now (did you add it?).')
    return get_refdoc(text_dir)



def next_word_probabilities(args, model, refdoc_id, split, prompt, top_n=10):
    """ Outputs the top n next word probabilities according to the model given a prompt and a refdoc. """
    data_module = SummarizationDataModule(args)
    data_loader = (
        data_module.train_dataloader() if split == 'train'
        else data_module.val_dataloader() if split == 'valid'
        else data_module.test_dataloader()
    )
    for batch in data_loader:
        if batch.refdoc[0] != refdoc_id:
            continue
        inputs = model.tokenizer(prompt, return_tensors='pt')
        batch.tgt = inputs['input_ids']
        batch.mask_tgt = inputs['attention_mask']
        with torch.no_grad():
            output = model(batch)
        probs = F.softmax(output.logits[0, -1], dim=-1)
        sorted_probs, indices = probs.sort(descending=True)
        print(f'Top {top_n} next word probabilities:')
        for i, (prob, idx) in enumerate(zip(sorted_probs, indices)):
            token = model.tokenizer.convert_ids_to_tokens([idx])[0]
            token = token.replace(BART_WHITESPACE_CHAR, '')
            print(f'{token:20s} - {prob:6.2%}')
            if i + 1 == top_n:
                break
        return
    print(f'Refdoc not found: {refdoc_id}')


def generate_completion(args, model, refdoc_id, split, prompt):
    data_module = SummarizationDataModule(args)
    data_loader = (
        data_module.train_dataloader() if split == 'train'
        else data_module.val_dataloader() if split == 'valid'
        else data_module.test_dataloader()
    )
    for batch in data_loader:
        if batch.refdoc[0] != refdoc_id:
            continue
        with torch.no_grad():
            source, reference, candidate = model.generate_with_prefix(batch, prompt)
        return source, reference, candidate
    print(f'Refdoc not found: {refdoc_id}')
    return None


def complete(args, model, refdoc_id, split, prompt):
    """ Completes the target prompt to the end given a refdoc. """
    result = generate_completion(args, model, refdoc_id, split, prompt)
    if result:
        _, _, candidate = result
        print(f'Completion: {candidate}')


def generate_full(args, model, refdoc_id, split):
    result = generate_completion(args, model, refdoc_id, split, None)
    if result:
        source, reference, candidate = result
        print(f'Source: {source}')
        print(f'Reference: {reference}')
        print(f'Candidate: {candidate}')


def main(args):
    seed_everything(args.seed)
    print('Loading BART model...')
    model = load_model(args)
    args.batch_size = 1
    refdoc_id, split = get_refdoc(args.text_dir)

    # mode: change [r]efdoc, [c]ompletion, [n]ext word, [f]ull generation, [e]xit
    while True:
        mode = None
        while mode not in ['r', 'f', 'c', 'n', 'e']:
            mode = input('Select an operation mode. Change [r]efdoc, generate a [f]ull interpretation,'
                         '[c]omplete a prefix, give [n]ext word probabilities, [e]xit: ')
        if mode == 'r':
            refdoc_id, split = get_refdoc(args.text_dir)
        elif mode == 'f':
            generate_full(args, model, refdoc_id, split)
        elif mode == 'c':
            prompt = input('Enter a prompt: ')
            prompt = prompt.lstrip()
            complete(args, model, refdoc_id, split, prompt)
        elif mode == 'n':
            prompt = input('Enter a prompt: ')
            prompt = prompt.lstrip()
            next_word_probabilities(args, model, refdoc_id, split, prompt, top_n=args.top_n)
        elif mode == 'e':
            break


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Runs BART interactively.')
    parser.add_argument('--dataset', default='fomc', choices=['fomc', 'us-russia'], help='Dataset name')
    parser.add_argument('--data_dir', default='data_fomc_bart', help='Path to preprocessed data dir')
    parser.add_argument('--text_dir', default='data_fomc_txt', help='Path to extracted text files')
    parser.add_argument('--model_dir', default='model_fomc', help='Path to finetuned model dir')
    parser.add_argument('--filter_model', default='filterbert', choices=['filterbert', 'oracle'],
                        help='Filtering model')
    parser.add_argument('--top_n', type=int, default=10, help='Top n next word probabilities to show')
    parser.add_argument('--seed', type=int, default=1, help='Random seed')

    # generation args
    parser.add_argument('--max_length', type=int, default=500, help='Max generation length')
    parser.add_argument('--min_length', type=int, default=50, help='Min generation length')
    parser.add_argument('--length_penalty', type=float, default=1.0, help='Alpha for length penalty')
    parser.add_argument('--ngram_blocking', type=int, default=0, help='Block repetition of n-grams (0: off)')

    main(parser.parse_args())
