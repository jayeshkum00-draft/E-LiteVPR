from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

# Column index as per the pairs.txt line
# (0 = timestamp, 1 = rgb)
_EVENT_COLUMN = {'histogram': 2, 'voxel': 3}

def sequence_name_from_rel_path(rel_path: str) -> str:
    """'rgb/zurich_city_03_a_000123.npy' -> 'zurich_city_03_a'"""
    return Path(rel_path).stem.rsplit('_', 1)[0]

class E_LiteVPRDataset(Dataset):
    """Dataset for E-LiteVPR training and evaluation."""
    def __init__(self,
                 root,
                 features_dir,
                 event_type: str = 'histogram',
                 sequences: Optional[Sequence[str]] = None,
                 pair_stride: int = 1):

        if event_type not in _EVENT_COLUMN:
            raise ValueError(
                f"Invalid event type: {event_type!r}."
                f"Must be one of {sorted(_EVENT_COLUMN)}."
            )              
        assert pair_stride >= 1          
                
        self.root = Path(root)
        self.features_dir = Path(features_dir)
        self.event_type = event_type
        event_col = _EVENT_COLUMN[event_type]

        pairs_file = self.root / 'pairs.txt'
        if not pairs_file.is_file():
            raise FileNotFoundError(
                f"{pairs_file} not found. Root should point towards the preprocessed dsec directory containing master pairs.txt"
                )
        
        sequences = set(sequences) if sequences is not None else None
        seen_sequences = set()
    
        self.pairs = []
        with open(pairs_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entries = line.split(',') # comma-separated
                assert len(entries) == 4, f"Expected 4 entries, got {len(entries)}"

                seq_name = sequence_name_from_rel_path(entries[1])
                seen_sequences.add(seq_name)
                if sequences is not None and seq_name not in sequences:
                    continue

                self.pairs.append({
                    'timestamp_us': int(entries[0]),
                    'sequence':seq_name,
                    # frames.txt in the cache stores 'seq_name/rgb/...' -- same join as cache script
                    'feature_key': f"{seq_name}/{entries[1]}",
                    'event_path': self.root / seq_name / entries[event_col],
                })

        if pair_stride > 1:
            self.pairs = self.pairs[::pair_stride]

        if len(self.pairs) == 0:
            raise RuntimeError(
                f"No pairs found in {pairs_file} after filtering by sequences={sequences}."
                f" Seen sequences: {sorted(seen_sequences)}"
            )
        
        # build (feature key -> row) index per sequence from frames.txt once
        self._row_index: dict = {}
        for seq in {pair['sequence'] for pair in self.pairs}:
            frames_file = self.features_dir / seq / 'frames.txt'
            if not frames_file.is_file():
                raise FileNotFoundError(
                    f"Expected frames.txt not found at {frames_file}. "
                    f"Check that the features_dir={self.features_dir} is correct."
                )
            with open(frames_file, 'r') as f:
                for row, line in enumerate(f):
                    line = line.strip()
                    if line:
                        self._row_index[line] = row
        
        # mmaps opened lazily {per dataloader worker} in _get_features
        self._mmaps: dict = {}
        
        # fail fast on a wrong root or unextracted dataset instead of mid-epoch
        probe = self.pairs[0]
        if not probe['event_path'].is_file():
            raise FileNotFoundError(
                f"Expected event file not found at {probe['event_path']}. "
                f"Check that the root={self.root} is correct and that the dataset has been preprocessed."
            )
        if probe['feature_key'] not in self._row_index:
            raise RuntimeError(
                f"Feature key {probe['feature_key']} not found in frames.txt index. "
                f"Check that the features_dir={self.features_dir} is correct and that the dataset has been preprocessed."
            )
        patches, attn = self._get_features(probe)
        assert patches.ndim == 2 and attn.ndim == 1, \
            f"Expected patches (N, D) and attn (N,), got {tuple(patches.shape)} and {tuple(attn.shape)}"
            
                
    def __len__(self):
        return len(self.pairs)
    
    def sequence_names(self):
        """Sorted sequence names present in this (filtered) dataset."""
        return sorted({pair['sequence'] for pair in self.pairs})
    
    def _load_event(self, event_path: Path) -> torch.Tensor:
        """Load the preprocessed event data stored as .npy"""
        event_data = np.load(event_path) # float16 (3, H, W), already normalized
        event_tensor = torch.from_numpy(event_data.astype(np.float32)) # convert to float32
        return event_tensor
    
    def _get_features(self, pair: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fetch cached teacher features (patches, attn) for a given pair from the memory-mapped .npy files."""
        seq = pair['sequence']
        if seq not in self._mmaps:
            seq_dir = self.features_dir / seq
            self._mmaps[seq] = (
                np.load(seq_dir / 'patches.npy', mmap_mode='r'),
                np.load(seq_dir / 'attn.npy', mmap_mode='r')
            )
        patches_mmap, attn_mmap = self._mmaps[seq]
        row = self._row_index[pair['feature_key']]
        patches = torch.from_numpy(np.ascontiguousarray(patches_mmap[row]).astype(np.float32))
        attn = torch.from_numpy(np.ascontiguousarray(attn_mmap[row]).astype(np.float32))
        return patches, attn
    
    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        pair = self.pairs[idx]
        event_tensor = self._load_event(pair['event_path'])
        teacher_patches, teacher_attn = self._get_features(pair)
        return event_tensor, teacher_patches, teacher_attn, pair['timestamp_us']