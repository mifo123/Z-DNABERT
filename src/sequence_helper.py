import numpy as np
from tqdm.auto import tqdm

class SequenceHelper:
    
    base_pair_opposite_map = {
        'A': 'T',
        'T': 'A',
        'C': 'G',
        'G': 'C',
    }

    def upper_seq(self, seq: str) -> str:
        return seq.upper()
    
    def complement_nucleobase(self, nucleobase: str) -> str:
        return self.base_pair_opposite_map[nucleobase] if nucleobase in self.base_pair_opposite_map else nucleobase
    
    def complement_seq(self, seq: str) -> str:
        return ''.join([self.complement_nucleobase(nucleobase) for nucleobase in seq])
    
    def reverse_seq(self, seq: str) -> str:
        return seq[::-1]
    
    def seq2kmer(self, seq: str, k: int):
        kmer = [seq[x:x+k] for x in range(len(seq)+1-k)]
        return kmer
    
    def split_seq(self, seq: str, length: int = 512, pad: int = 16):
        res = []
        for st in range(0, len(seq), length - pad):
            end = min(st+length, len(seq))
            res.append(seq[st:end])
        return res

    def stitch_np_preds_slow(
        self,
        np_seqs,
        progress_bar=tqdm,
        pad=16,
    ):
        res = np.array([])
        for seq in progress_bar(np_seqs, 'stitching predictions'):
            res = res[:-pad]
            res = np.concatenate([res, seq])
        return res


    def stitch_np_preds(
            self,
            np_seqs,
            progress_bar=tqdm,
            pad=16,
    ):
        if not np_seqs:
            return np.array([], dtype=np.float32)

        # 1) Spočítaj výslednú dĺžku presne podľa pôvodného algoritmu:
        #    res = res[:-pad]; res = concat([res, seq])
        lengths = [len(seq) for seq in np_seqs]

        # predalokácia podľa teórie (ako doteraz)
        total_len = 0
        for i, l in enumerate(lengths):
            if i == 0:
                # prvý kus: res = concat([], seq) -> len = l
                total_len = l
            else:
                # res = res[:-pad]
                reduced = total_len - pad
                if reduced < 0:
                    reduced = 0
                # res = concat(res, seq)
                total_len = reduced + l

        # 2) Prealokuj výstup
        dtype = np_seqs[0].dtype
        res = np.empty(total_len, dtype=dtype)

        # 3) Druhé prečítanie – zapisuj na správne offsety
        pos = 0
        first = True

        for seq in progress_bar(np_seqs, 'stitching predictions'):
            l = len(seq)
            if first:
                if l > res.shape[0]:
                    res = np.empty(l, dtype=dtype)
                res[0:l] = seq
                pos = l
                first = False
                continue

            new_pos = pos - pad
            if new_pos < 0:
                new_pos = 0

            end = new_pos + l

            # 🔥 kľúč: ak predalokácia nesedí, zväčši res (zriedkavé, ale zachráni job)
            if end > res.shape[0]:
                new_res = np.empty(end, dtype=dtype)
                new_res[:pos] = res[:pos]
                res = new_res

            res[new_pos:end] = seq
            pos = end

        return res[:pos]

