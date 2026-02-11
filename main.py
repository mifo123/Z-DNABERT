import os
import pathlib
import argparse
import time
import traceback

from tqdm.auto import tqdm

from src.prediction_runner import PredictionRunner
from src.prediction_input import PredictionInput
from src.prediction_input_file_from_filesystem import PredictionInputFileFromFilesystem
from src.zdnabert_model import ZdnabertModel
from src.sequence_variation_normal import SequenceVariationNormal
from src.sequence_variation_reverse_complement import SequenceVariationReverseComplement
from src.prediction_result_formatter_bed_file import PredictionResultFormatterBedFile
from src.zdnabert_model_downloader import ZdnabertModelDownloader
from src.status_file import update_status_file

MODEL_DOWNLOAD_PATH = "./pytorch_models"
INPUT_PATH = "./input"
OUTPUT_PATH = "./output"


# -----------------------------
# Torch / BLAS nastavenie threadov
# -----------------------------
try:
    import torch

    NUM_THREADS = int(os.getenv("ZDNABERT_NUM_THREADS", "8"))

    # počet CPU threadov pre výpočty
    torch.set_num_threads(NUM_THREADS)
    # interop thready nechajme nízke, väčšinou 1 stačí
    torch.set_num_interop_threads(1)
except ImportError:
    # ak by torch nebol, nechceme padať pri importe
    pass


# -----------------------------
# Lazy init modelov (download len raz na proces)
# -----------------------------
_model_downloader = ZdnabertModelDownloader()
_models_ready = False


def ensure_models_ready() -> None:
    global _models_ready
    if _models_ready:
        return
    _model_downloader.download_models(MODEL_DOWNLOAD_PATH)
    _model_downloader.download_metas(MODEL_DOWNLOAD_PATH)
    _models_ready = True


def run_zdnabert_analysis(
    model: str,
    input_fasta: pathlib.Path,
    confidence_threshold: float = 0.5,
    min_seq_length: int = 10,
    check_reverse_complement: bool = False,
    use_cuda: bool = False,
    status_path: pathlib.Path | None = None,
    sequence_length: int | None = None,
) -> list[str]:
    # zaisti, že modely sú stiahnuté (len prvýkrát v procese)
    ensure_models_ready()

    zdnabert_model = ZdnabertModel(
        os.path.join(MODEL_DOWNLOAD_PATH, model),
        model_name=model,
        model_confidence_threshold=confidence_threshold,
        minimum_sequence_length=min_seq_length,
        use_cuda=use_cuda,
    )
    zdnabert_model.load()

    sequence_variations = [SequenceVariationNormal()]
    if check_reverse_complement:
        sequence_variations.append(SequenceVariationReverseComplement())

    if not input_fasta.is_file():
        raise FileNotFoundError(f"{input_fasta} is not a file.")

    prediction_input_file = PredictionInputFileFromFilesystem(
        input_fasta.name, input_fasta
    )

    prediction_input = PredictionInput(
        zdnabert_model,
        [prediction_input_file],
        sequence_variations,
    )

    prediction_runner = PredictionRunner()
    formatter = PredictionResultFormatterBedFile()

    # vyber progress bar podľa toho, či máme status_path
    if status_path is not None:
        progress_bar = make_status_progress_bar(status_path)
    else:
        progress_bar = tqdm

    results: list[str] = []
    print("[DEBUG] Starting prediction_runner.run()")
    for prediction_result in prediction_runner.run([prediction_input], progress_bar):
        print(
            f"[DEBUG] Got PredictionResult "
            f"file={prediction_result.file_name} "
            f"seq={prediction_result.seq_record_name} "
            f"len={prediction_result.seq_len}"
        )
        print("[DEBUG] Entering formatter.format()")
        count = 0
        for line in formatter.format(prediction_result):
            results.append(line)
            count += 1
        print(f"[DEBUG] formatter.format() returned {count} lines")
        print("[DEBUG] Finished prediction_runner.run() / formatting")

    return results

def _normalize_stage_name(desc: str | None) -> str:
    if not desc:
        return "unknown"

    d = desc.lower()
    if "prediction on sequence pieces" in d:
        return "prediction_pieces"
    if "stitching" in d:
        return "stitching"
    if "records" in d:
        return "records"
    if "sequences" in d:
        return "sequences"
    if "files" in d:
        return "files"
    if "inputs" in d:
        return "inputs"
    return desc


def make_status_progress_bar(status_path: pathlib.Path) -> callable:
    """
    Vráti funkciu kompatibilnú s tqdm, ktorá:
      - vytvorí tqdm progress bar
      - pri update() raz za pár sekúnd zapíše progress/ETA do status súboru
    """
    MIN_INTERVAL = 5.0
    last_write = {"t": 0.0}

    def progress_bar(iterable, desc=None, *args, **kwargs):
        pbar = tqdm(iterable, desc=desc, *args, **kwargs)

        if desc is None:
            return pbar

        stage = _normalize_stage_name(desc)
        orig_update = pbar.update

        def update(n=1):
            res = orig_update(n)
            now = time.time()
            if now - last_write["t"] >= MIN_INTERVAL:
                try:
                    total = pbar.total or 0
                    current = pbar.n
                    progress = float(current) / total if total else None

                    # ETA z rate (iterácie/s): (total - n) / rate
                    fd = getattr(pbar, "format_dict", {}) or {}
                    rate = fd.get("rate", None)
                    remaining = None
                    if (
                        total
                        and current is not None
                        and rate is not None
                        and rate > 0
                    ):
                        remaining = (total - current) / rate

                    patch = {
                        "status": "running",
                        "stage": stage,
                        "progress": progress,
                        "eta_seconds": remaining,
                    }
                    update_status_file(status_path, patch)
                except Exception:
                    traceback.print_exc()
                last_write["t"] = now
            return res

        pbar.update = update
        return pbar

    return progress_bar


def main():
    parser = argparse.ArgumentParser(
        description="Run Z-DNABERT predictions on input sequences."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["HG_chipseq", "HG_kouzine", "MM_curax", "MM_kouzine"],
        help="Name of the model to use. Choices are: HG_chipseq, HG_kouzine, MM_curax, MM_kouzine.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Model confidence threshold.",
    )
    parser.add_argument(
        "--min-seq-length",
        type=int,
        default=10,
        help="Minimum sequence length to process.",
    )
    parser.add_argument(
        "--check-reverse-complement",
        action="store_true",
        help="Check reverse complement sequence variations.",
    )
    parser.add_argument(
        "--use-cuda", action="store_true", help="Use CUDA if available."
    )
    parser.add_argument(
        "--output", type=str, default=OUTPUT_PATH, help="Path to save output files."
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input", type=str, help="Path to directory with input FASTA files."
    )
    input_group.add_argument(
        "--input-file", type=str, help="Path to a single FASTA input file."
    )

    args = parser.parse_args()

    output_path = pathlib.Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    sequence_variations = [SequenceVariationNormal()]
    if args.check_reverse_complement:
        sequence_variations.append(SequenceVariationReverseComplement())

    prediction_input_files: list[PredictionInputFileFromFilesystem] = []

    if args.input_file:
        input_file_path = pathlib.Path(args.input_file)
        if not input_file_path.is_file():
            raise FileNotFoundError(
                f"Input file '{input_file_path}' does not exist or is not a file."
            )
        prediction_input_files.append(
            PredictionInputFileFromFilesystem(
                input_file_path.name, input_file_path
            )
        )
    else:
        input_dir_path = pathlib.Path(args.input)
        if not input_dir_path.is_dir():
            raise NotADirectoryError(
                f"Input path '{input_dir_path}' is not a directory."
            )
        for file_path in input_dir_path.iterdir():
            if file_path.is_file():
                prediction_input_files.append(
                    PredictionInputFileFromFilesystem(
                        file_path.name, file_path
                    )
                )

    if not prediction_input_files:
        raise ValueError("No input files found.")

    # zaisti, že modely sú stiahnuté (len raz)
    ensure_models_ready()

    zdnabert_model = ZdnabertModel(
        os.path.join(MODEL_DOWNLOAD_PATH, args.model),
        model_name=args.model,
        model_confidence_threshold=args.confidence_threshold,
        minimum_sequence_length=args.min_seq_length,
        use_cuda=args.use_cuda,
    )

    prediction_input = PredictionInput(
        zdnabert_model,
        prediction_input_files,
        sequence_variations,
    )

    prediction_inputs = [prediction_input]
    prediction_result_formatter_bed_file = PredictionResultFormatterBedFile()
    prediction_runner = PredictionRunner()

    now_time_as_string_for_file_name = time.strftime("%Y_%m_%d,%H_%M_%S")
    for prediction_result in prediction_runner.run(
        prediction_inputs, progress_bar=tqdm
    ):
        bed_file_name = prediction_result_formatter_bed_file.file_name_variation(
            prediction_result, now_time_as_string_for_file_name
        )
        output_file_path = output_path / bed_file_name

        with open(output_file_path, "w") as bed_file:
            for line in prediction_result_formatter_bed_file.format(
                prediction_result
            ):
                bed_file.write(f"{line}\n")

        print(f"Results saved to {output_file_path}")


if __name__ == "__main__":
    main()
