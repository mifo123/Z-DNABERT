import os
import pathlib
import argparse
import time
from tqdm.auto import tqdm
from src.prediction_runner import PredictionRunner
from src.prediction_input import PredictionInput
from src.prediction_input_file_from_filesystem import PredictionInputFileFromFilesystem
from src.zdnabert_model import ZdnabertModel
from src.sequence_variation_normal import SequenceVariationNormal
from src.sequence_variation_reverse_complement import SequenceVariationReverseComplement
from src.prediction_result_formatter_bed_file import PredictionResultFormatterBedFile
from src.zdnabert_model_downloader import ZdnabertModelDownloader

MODEL_DOWNLOAD_PATH = './pytorch_models'
INPUT_PATH = './input'
OUTPUT_PATH = './output'


def run_zdnabert_analysis(
    model: str,
    input_fasta: pathlib.Path,
    confidence_threshold: float = 0.5,
    min_seq_length: int = 10,
    check_reverse_complement: bool = False,
    use_cuda: bool = False,
) -> list[str]:
    zdnabert_model_downloader = ZdnabertModelDownloader()
    zdnabert_model_downloader.download_models(MODEL_DOWNLOAD_PATH)
    zdnabert_model_downloader.download_metas(MODEL_DOWNLOAD_PATH)

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

    prediction_input_file = PredictionInputFileFromFilesystem(input_fasta.name, input_fasta)

    prediction_input = PredictionInput(
        zdnabert_model,
        [prediction_input_file],
        sequence_variations,
    )

    prediction_runner = PredictionRunner()
    formatter = PredictionResultFormatterBedFile()

    results = []
    print("[DEBUG] Starting prediction_runner.run()")
    for prediction_result in prediction_runner.run([prediction_input]):
        print(f"[DEBUG] Got PredictionResult "
              f"file={prediction_result.file_name} "
              f"seq={prediction_result.seq_record_name} "
              f"len={prediction_result.seq_len}")
        print("[DEBUG] Entering formatter.format()")
        bed_chunk = list(formatter.format(prediction_result))
        print(f"[DEBUG] formatter.format() returned {len(bed_chunk)} lines")
        results.extend(formatter.format(prediction_result))
        print("[DEBUG] Finished prediction_runner.run() / formatting")

    return results


def main():
    parser = argparse.ArgumentParser(description="Run Z-DNABERT predictions on input sequences.")
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        choices=["HG_chipseq", "HG_kouzine", "MM_curax", "MM_kouzine"],
        help="Name of the model to use. Choices are: HG_chipseq, HG_kouzine, MM_curax, MM_kouzine."
    )

    parser.add_argument('--confidence-threshold', type=float, default=0.5, help="Model confidence threshold.")
    parser.add_argument('--min-seq-length', type=int, default=10, help="Minimum sequence length to process.")
    parser.add_argument('--check-reverse-complement', action='store_true', help="Check reverse complement sequence variations.")
    parser.add_argument('--use-cuda', action='store_true', help="Use CUDA if available.")
    parser.add_argument('--output', type=str, default=OUTPUT_PATH, help="Path to save output files.")

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--input', type=str, help="Path to directory with input FASTA files.")
    input_group.add_argument('--input-file', type=str, help="Path to a single FASTA input file.")

    args = parser.parse_args()

    output_path = pathlib.Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    sequence_variations = [SequenceVariationNormal()]
    if args.check_reverse_complement:
        sequence_variations.append(SequenceVariationReverseComplement())

    prediction_input_files = []

    if args.input_file:
        input_file_path = pathlib.Path(args.input_file)
        if not input_file_path.is_file():
            raise FileNotFoundError(f"Input file '{input_file_path}' does not exist or is not a file.")
        prediction_input_files.append(
            PredictionInputFileFromFilesystem(input_file_path.name, input_file_path)
        )
    else:
        input_dir_path = pathlib.Path(args.input)
        if not input_dir_path.is_dir():
            raise NotADirectoryError(f"Input path '{input_dir_path}' is not a directory.")
        for file_path in input_dir_path.iterdir():
            if file_path.is_file():
                prediction_input_files.append(
                    PredictionInputFileFromFilesystem(file_path.name, file_path)
                )

    if not prediction_input_files:
        raise ValueError("No input files found.")

    zdnabert_model_downloader = ZdnabertModelDownloader()
    zdnabert_model_downloader.download_models(MODEL_DOWNLOAD_PATH)
    zdnabert_model_downloader.download_metas(MODEL_DOWNLOAD_PATH)

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
    for prediction_result in prediction_runner.run(prediction_inputs, progress_bar=tqdm):
        bed_file_name = prediction_result_formatter_bed_file.file_name_variation(prediction_result, now_time_as_string_for_file_name)
        output_file_path = output_path / bed_file_name

        with open(output_file_path, 'w') as bed_file:
            for line in prediction_result_formatter_bed_file.format(prediction_result):
                bed_file.write(f"{line}\n")

        print(f"Results saved to {output_file_path}")

if __name__ == "__main__":
    main()
