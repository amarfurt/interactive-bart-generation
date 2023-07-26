""" Data module, batch and dataset. """

import glob
import os

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset


dep_labels = {'PAD':0, 'UNK':1,'main_root':2, 'nsubjpass':3, 'mwe':4, 'discourse':5, 'advcl':6, 'quantmod':7, 'predet':8, 'iobj':9,
              'pobj':10, 'cop':11, 'punct':12, 'num':13,'ccomp':14, 'xcomp':15, 'poss':16, 'preconj':17, 'csubjpass':18,
              'parataxis':19, 'rcmod':20, 'npadvmod':21, 'expl':22, 'advmod':23, 'nsubj':24, 'amod':25, 'prep':26,
              'aux':27, 'prt':28, 'det':29, 'dep':30, 'partmod':31, 'root':32, 'infmod':33, 'mark':34, 'number':35,
              'appos':36, 'possessive':37, 'csubj':38, 'acomp':39, 'neg':40, 'conj':41, 'auxpass':42, 'nn':43,
              'tmod':44, 'dobj':45, 'cc':46, 'pcomp':47, 'crop': 48}

srl_labels = {'PAD': 0, 'UNK': 1, 'ARGM-TMP': 2, 'ARG0': 3, 'V': 4, 'ARG1': 5, 'ARG2': 6, 'ARGM-PRD': 7, 'ARG3': 8,
              'ARGM-ADV': 9, 'ARGM-MOD': 10, 'ARGM-DIS': 11, 'ARGM-CAU': 12, 'ARGM-MNR': 13, 'ARGM-PRP': 14,
              'R-ARG1': 15, 'ARGM-DIR': 16, 'C-ARG1': 17, 'ARGM-LOC': 18, 'ARG4': 19, 'ARGM-NEG': 20, 'C-ARG0': 21,
              'R-ARG0': 22, 'ARGM-EXT': 23, 'C-ARG2': 24, 'C-ARGM-MNR': 25, 'R-ARGM-TMP': 26, 'ARGM-ADJ': 27,
              'R-ARGM-MNR': 28, 'R-ARG2': 29, 'R-ARGM-LOC': 30, 'ARGM-PNC': 31, 'ARG5': 32, 'ARGM-REC': 33,
              'ARGM-GOL': 34, 'R-ARGM-EXT': 35, 'ARGM-COM': 36, 'R-ARGM-CAU': 37, 'C-ARGM-EXT': 38, 'ARGM-LVB': 39,
              'C-ARGM-ADV': 40, 'R-ARG3': 41, 'C-ARG4': 42, 'R-ARGM-ADV': 43, 'crop': 44}


class SummarizationDataModule(LightningDataModule):

    def __init__(self, args):
        super().__init__()
        self.data_dir = args.data_dir
        self.dataset = args.dataset
        self.data_model = 'bart' if args.model == 'bart' else 'bert'
        self.filter_model = args.filter_model
        self.num_workers = args.num_workers
        self.batch_size_train = args.batch_size
        self.batch_size_test = 1  # .generate fuses examples when batch size > 1
        self.max_pos = 512 if self.data_model == 'bert' else 1024
        self.max_tgt_len = 512
        self.tgt_eos_id = 2

    def _truncate_bert(self, x):
        x['src'] = x['src'][:-1][:self.max_pos - 1] + x['src'][-1:]  # slicing notation works with empty inputs
        x['tgt'] = x['tgt'][:self.max_tgt_len][:-1] + [self.tgt_eos_id]
        x['src_segs'] = x['src_segs'][:self.max_pos]

        if 'heads' in x.keys():
            x['heads'] = list(x['heads'])
            x['rels'] = list(x['rels'])
            x['heads'] = x['heads'][:-1][:self.max_pos - 1] + x['heads'][-1:]
            # x['rels'] = [dep_labels[x] for x in x['rels']]
            x['rels'] = x['rels'][:-1][:self.max_pos - 1] + x['rels'][-1:]

            # x['heads'] = [min(y,len(x['src'])-1) for y in x['heads']]
            for i, head in enumerate(x['heads']):
                if head >= len(x['src']):
                    x['heads'][i] = 0
                    x['rels'][i] = dep_labels['crop']
            if x['heads'][-1] == len(x['src']) - 1:
                x['heads'][-1] = 0
                x['rels'][-1] = dep_labels['crop']

            assert max(x['heads']) < len(x['src']), x['heads']

        if "srl_list" in x.keys():
            x['base_vec'] = x['base_vec'][:self.max_pos]
            x['rearrange_ids'] = x['rearrange_ids'][:self.max_pos]
            x['srl_list'] = x['srl_list']

        return x

    def _truncate_bart(self, x):
        x['src'] = x['src'][:self.max_pos][:-1] + x['src'][-1:]
        x['tgt'] = x['tgt'][:self.max_tgt_len][:-1] + x['tgt'][-1:]
        return x

    def collate(self, data):
        assert self.data_model in ['bert', 'bart'], f"Unknown data model: {self.data_model}"
        if self.data_model == 'bert':
            data = list(map(self._truncate_bert, data))
            return SummarizationBatch(data)
        else:
            return BartBatch(list(map(self._truncate_bart, data)))

    def train_dataloader(self):
        dataset = SummarizationDataset(
            data_dir=self.data_dir,
            dataset=self.dataset,
            filter_model=self.filter_model,
            split='train',
        )

        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size_train,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.collate,
            pin_memory=True,
        )

    def val_dataloader(self):
        dataset = SummarizationDataset(
            data_dir=self.data_dir,
            dataset=self.dataset,
            filter_model=self.filter_model,
            split='valid',
        )
        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size_test,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate,
            pin_memory=True,
        )

    def test_dataloader(self):
        dataset = SummarizationDataset(
            data_dir=self.data_dir,
            dataset=self.dataset,
            filter_model=self.filter_model,
            split='test',
        )
        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size_test,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate,
            pin_memory=True,
        )


class SummarizationBatch:

    def __init__(self, data, pad_id=0):
        self.batch_size = len(data)
        self.pad_id = pad_id
        self.src = torch.tensor(self.pad([x['src'] for x in data]))
        self.tgt = torch.tensor(self.pad([x['tgt'] for x in data]))
        self.segs = torch.tensor(self.pad([x['src_segs'] for x in data]))

        if 'heads' in data[0].keys():
            self.graph_syn = True
            self.heads = torch.tensor(self.pad([x['heads'] for x in data]))
            self.rels = torch.tensor(self.pad([x['rels'] for x in data]))
        else:
            self.graph_syn = False

        if 'srl_list' in data[0].keys():
            self.graph_srl = True
            self.base_vecs = torch.tensor(self.pad([x['base_vec'] for x in data]))
            self.rearrange_ids = torch.tensor(self.pad([x['rearrange_ids'] for x in data]))
            self.srl_list = [x['srl_list'] for x in data]
        else:
            self.graph_srl = False

        self.mask_src = 1 - (self.src == 0).to(torch.uint8)
        self.mask_tgt = 1 - (self.tgt == 0).to(torch.uint8)
        self.refdoc = [x['name'] for x in data]

    def pad(self, data):
        """ Pad `data` to same length with `pad_id`. """
        max_len = max(len(x) for x in data)
        return [x + [self.pad_id] * (max_len - len(x)) for x in data]

    def __len__(self):
        return self.batch_size

    def to(self, *args, **kwargs):
        self.src = self.src.to(*args, **kwargs)
        self.tgt = self.tgt.to(*args, **kwargs)

        if self.graph_syn:
            self.heads = self.heads.to(*args, **kwargs)
            self.rels = self.rels.to(*args, **kwargs)

        if self.graph_srl:
            self.base_vecs = self.base_vecs.to(*args, **kwargs)
            self.rearrange_ids = self.rearrange_ids.to(*args, **kwargs)

        self.segs = self.segs.to(*args, **kwargs)
        self.mask_src = self.mask_src.to(*args, **kwargs)
        self.mask_tgt = self.mask_tgt.to(*args, **kwargs)
        return self

    def pin_memory(self):
        self.src = self.src.pin_memory()
        self.tgt = self.tgt.pin_memory()

        if self.graph_syn:
            self.heads = self.heads.pin_memory()
            self.rels = self.rels.pin_memory()

        if self.graph_srl:
            self.base_vecs = self.base_vecs.pin_memory()
            self.rearrange_ids = self.rearrange_ids.pin_memory()

        self.segs = self.segs.pin_memory()
        self.mask_src = self.mask_src.pin_memory()
        self.mask_tgt = self.mask_tgt.pin_memory()
        return self


class BartBatch:

    def __init__(self, data, pad_id=1):
        self.batch_size = len(data)
        self.pad_id = pad_id
        self.src = torch.tensor(self.pad([x['src'] for x in data]))
        self.tgt = torch.tensor(self.pad([x['tgt'] for x in data]))
        self.mask_src = 1 - (self.src == pad_id).to(torch.uint8)
        self.mask_tgt = 1 - (self.tgt == pad_id).to(torch.uint8)
        self.refdoc = [x['name'] for x in data]

    def pad(self, data):
        """ Pad `data` to same length with `pad_id`. """
        max_len = max(len(x) for x in data)
        return [x + [self.pad_id] * (max_len - len(x)) for x in data]

    def __len__(self):
        return self.batch_size

    def to(self, *args, **kwargs):
        self.src = self.src.to(*args, **kwargs)
        self.tgt = self.tgt.to(*args, **kwargs)
        self.mask_src = self.mask_src.to(*args, **kwargs)
        self.mask_tgt = self.mask_tgt.to(*args, **kwargs)
        return self

    def pin_memory(self):
        self.src = self.src.pin_memory()
        self.tgt = self.tgt.pin_memory()
        self.mask_src = self.mask_src.pin_memory()
        self.mask_tgt = self.mask_tgt.pin_memory()
        return self


class SummarizationDataset(Dataset):

    def __init__(self, data_dir, dataset, filter_model, split='train'):
        data_files = sorted(glob.glob(os.path.join(data_dir, f'{dataset}.{filter_model}.{split}.pt')))
        self.data = []
        for pt in data_files:
            self.data.extend(torch.load(pt))

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
