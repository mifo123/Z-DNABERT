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

    def load(self) -> None:
        self.prepare_bert_model()
        self.check_cuda()
        self.prepare_device()

    def prepare_bert_model(self) -> None:
        self.tokenizer = BertTokenizer.from_pretrained(self.data_path)
        self.model = BertForTokenClassification.from_pretrained(self.data_path)

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

    def run_prediction(
        self,
        seq_pieces: list,
        progress_bar = tqdm,
    ) -> list:
        preds = []
        self.model.eval()
        with torch.no_grad():
            for seq_piece in progress_bar(seq_pieces, 'prediction on sequence pieces'):
                input_ids = torch.LongTensor(self.tokenizer.encode(' '.join(seq_piece), add_special_tokens=False)).to(self.device)
                input_ids_unsqueezed = input_ids.unsqueeze(0)  # shape [1, L]
                outputs = self.model(input_ids_unsqueezed)
                logits = outputs.logits  # shape [1, L, num_labels]
                probs = torch.softmax(logits, dim=-1)[0, :, 1]
                preds.append(probs.detach().cpu().numpy())
        return preds

    def stitch_preds(
        self,
        preds: list,
        progress_bar = tqdm,
    ):
        return self.sequence_helper.stitch_np_preds(preds, progress_bar)
    
    def label_stitched_preds(self, stitched_preds):
        labeled, max_label = scipy.ndimage.label(stitched_preds>self.model_confidence_threshold)
        return labeled, max_label