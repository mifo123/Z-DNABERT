import logging
from tqdm.auto import tqdm
import torch
import scipy
from transformers import BertTokenizer, BertForTokenClassification, PreTrainedModel
from src.sequence_helper import SequenceHelper

class ZdnabertModel:
    logger: logging.Logger
    using_cuda: bool
    tokenizer: BertTokenizer
    model: PreTrainedModel

    def __init__(
        self,
        data_path: str,
        model_name: str,
        model_confidence_threshold: float,
        minimum_sequence_length: int,
        use_cuda: bool,
    ):
        self.logger = logging.getLogger(__name__)
        self.sequence_helper = SequenceHelper()

        self.data_path = data_path
        self.model_name = model_name
        self.model_confidence_threshold = model_confidence_threshold
        self.minimum_sequence_length = minimum_sequence_length
        self.use_cuda = use_cuda

        self.using_cuda = False
        self.device = torch.device("cpu")  # default

        # pre verziu B – mapovanie k-mer -> token_id
        self.kmer_to_id: dict[str, int] | None = None

    def load(self) -> None:
        self.prepare_bert_model()
        self.check_cuda()
        self.prepare_device()

    def prepare_bert_model(self) -> None:
        self.tokenizer = BertTokenizer.from_pretrained(self.data_path)
        self.model = BertForTokenClassification.from_pretrained(self.data_path)

        # pre verziu B – predpočítaj vocab (token -> id)
        # pri typickom DNA k-mer BERT-e sú k-mery priamo tokeny vo vocab-e
        self.kmer_to_id = self.tokenizer.get_vocab()

    def check_cuda(self) -> None:
        if self.use_cuda and torch.cuda.is_available():
            self.using_cuda = True
            self.device = torch.device("cuda")
            self.logger.info("Using CUDA for ZDNABERT model.")
        else:
            if self.use_cuda and not torch.cuda.is_available():
                self.logger.warning(
                    "CUDA was requested but is not available. Falling back to CPU."
                )
            self.using_cuda = False
            self.device = torch.device("cpu")
            self.logger.info("Using CPU for ZDNABERT model.")

    def prepare_device(self) -> None:
        self.model.to(self.device)

    def kmer_and_split_seq(self, seq: str) -> list:
        kmer_seq = self.sequence_helper.seq2kmer(seq, 6)
        seq_pieces = self.sequence_helper.split_seq(kmer_seq)
        return seq_pieces

    # -------------------------------------------------------------------------
    # C) Originálna verzia – 1 sekvencia = 1 forward, tokenizer.encode v cykle
    # -------------------------------------------------------------------------
    def run_prediction_orig(
        self,
        seq_pieces: list,
        progress_bar=tqdm,
    ) -> list:
        """
        C) Originál – bez batchingu, tokenizácia encode() v cykle.
        """
        preds: list = []
        self.model.eval()
        with torch.inference_mode():
            for seq_piece in progress_bar(
                seq_pieces,
                'prediction on sequence pieces (orig)',
            ):
                # seq_piece: list[str] (k-mery)
                text = ' '.join(seq_piece)
                input_ids = torch.LongTensor(
                    self.tokenizer.encode(text, add_special_tokens=False)
                ).to(self.device)

                input_ids_unsqueezed = input_ids.unsqueeze(0)  # [1, L]
                outputs = self.model(input_ids_unsqueezed)
                logits = outputs.logits  # [1, L, num_labels]
                probs = torch.softmax(logits, dim=-1)[0, :, 1]
                preds.append(probs.cpu().numpy())
        return preds

    # -------------------------------------------------------------------------
    # A) Batched + batch HuggingFace tokenizér
    # -------------------------------------------------------------------------
    def run_prediction_batched(
        self,
        seq_pieces: list,
        progress_bar=tqdm,
        batch_size: int = 16,
    ) -> list:
        """
        A) Batched predikcia s HuggingFace tokenizérom, ale v batchi.
        seq_pieces: list[list[str]] (k-mery)
        """
        preds: list = []

        # priprava textov pre tokenizer len raz
        texts = [' '.join(piece) for piece in seq_pieces]

        self.model.eval()
        with torch.inference_mode():
            for start in progress_bar(
                range(0, len(texts), batch_size),
                'prediction on sequence pieces (batched)',
            ):
                batch_texts = texts[start:start + batch_size]
                if not batch_texts:
                    break

                enc = self.tokenizer(
                    batch_texts,
                    add_special_tokens=False,
                    padding=True,
                    return_tensors='pt',
                )

                input_ids = enc['input_ids'].to(self.device)            # [B, L]
                attention_mask = enc['attention_mask'].to(self.device)  # [B, L]

                outputs = self.model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits  # [B, L, num_labels]
                probs = torch.softmax(logits, dim=-1)[:, :, 1]  # [B, L]

                probs_np = probs.cpu().numpy()
                mask_np = attention_mask.cpu().numpy()

                # rozpad batche na jednotlivé sekvencie bez paddingu
                for i in range(probs_np.shape[0]):
                    valid_len = int(mask_np[i].sum())
                    preds.append(probs_np[i, :valid_len])

        return preds

    # -------------------------------------------------------------------------
    # B) Vlastná mapovacia tokenizácia k-merov (bez HF tokenizer v runtime)
    # -------------------------------------------------------------------------
    def run_prediction_kmer_ids(
        self,
        seq_pieces: list,
        progress_bar=tqdm,
        batch_size: int = 16,
    ) -> list:
        """
        B) Batched predikcia s vlastnou k-mer -> token_id mapou.
        Nepoužíva HF tokenizér v cykle.
        seq_pieces: list[list[str]] (k-mery)
        """
        if self.kmer_to_id is None:
            # mala by byť nastavená v prepare_bert_model, ale pre istotu
            self.kmer_to_id = self.tokenizer.get_vocab()

        # fallback hodnoty
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = 0
        unk_id = self.tokenizer.unk_token_id
        if unk_id is None:
            unk_id = 0

        preds: list = []

        self.model.eval()
        with torch.inference_mode():
            for start in progress_bar(
                range(0, len(seq_pieces), batch_size),
                'prediction on sequence pieces (batched, kmer ids)',
            ):
                batch_pieces = seq_pieces[start:start + batch_size]
                batch_size_eff = len(batch_pieces)
                if batch_size_eff == 0:
                    break

                # max dĺžka sekvencie v batche kvôli paddingu
                max_len = max(len(p) for p in batch_pieces)

                # [B, L] input_ids a attention_mask
                input_ids = torch.full(
                    (batch_size_eff, max_len),
                    fill_value=pad_id,
                    dtype=torch.long,
                    device=self.device,
                )
                attention_mask = torch.zeros(
                    (batch_size_eff, max_len),
                    dtype=torch.long,
                    device=self.device,
                )

                # naplnenie batchu – mapovanie k-mer -> id
                for i, piece in enumerate(batch_pieces):
                    ids = [self.kmer_to_id.get(k, unk_id) for k in piece]
                    seq_len = len(ids)
                    input_ids[i, :seq_len] = torch.tensor(
                        ids,
                        dtype=torch.long,
                        device=self.device,
                    )
                    attention_mask[i, :seq_len] = 1

                outputs = self.model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits  # [B, L, num_labels]
                probs = torch.softmax(logits, dim=-1)[:, :, 1]  # [B, L]

                probs_np = probs.cpu().numpy()
                mask_np = attention_mask.cpu().numpy()

                for i in range(batch_size_eff):
                    valid_len = int(mask_np[i].sum())
                    preds.append(probs_np[i, :valid_len])

        return preds

    # -------------------------------------------------------------------------
    # Wrapper – unified public API, ľahké prepínanie režimov
    # -------------------------------------------------------------------------
    def run_prediction(
        self,
        seq_pieces: list,
        progress_bar=tqdm,
        batch_size: int = 16,
        mode: str = "kmer",
    ) -> list:
        """
        Unified wrapper, aby sa dali jednoducho prepínať režimy.

        mode: "orig" | "batched" | "kmer"
        """
        if mode == "orig":
            return self.run_prediction_orig(
                seq_pieces,
                progress_bar=progress_bar,
            )
        elif mode == "batched":
            return self.run_prediction_batched(
                seq_pieces,
                progress_bar=progress_bar,
                batch_size=batch_size,
            )
        elif mode == "kmer":
            return self.run_prediction_kmer_ids(
                seq_pieces,
                progress_bar=progress_bar,
                batch_size=batch_size,
            )
        else:
            raise ValueError(f"Unknown prediction mode: {mode}")

    # -------------------------------------------------------------------------
    # Stitching a labeling
    # -------------------------------------------------------------------------
    def stitch_preds(
        self,
        preds: list,
        progress_bar=tqdm,
    ):
        # predpoklad: SequenceHelper má optimalizovanú stitch_np_preds
        return self.sequence_helper.stitch_np_preds(preds, progress_bar)

    def label_stitched_preds(self, stitched_preds):
        labeled, max_label = scipy.ndimage.label(
            stitched_preds > self.model_confidence_threshold
        )
        return labeled, max_label
