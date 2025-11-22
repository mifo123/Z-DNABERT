from typing import Iterable
import numpy as np
from src.prediction_result_formatter import PredictionResultFormatter
from src.prediction_result import PredictionResult

class PredictionResultFormatterBedFile(PredictionResultFormatter):
    def file_name_common(self, prediction_result: PredictionResult, now_time_as_string_for_file_name: str) -> str:
        model_params_as_string_for_file_name = prediction_result.get_model_params_as_string_for_file_name()
        file_name = prediction_result.file_name
        seq_record_name = prediction_result.seq_record_name
        seq_record_key = '{}.{}.{}.{}'.format(file_name, seq_record_name, model_params_as_string_for_file_name, now_time_as_string_for_file_name)
        bed_file_name = '{}.bed'.format(seq_record_key)

        return bed_file_name
    
    def file_name_variation(self, prediction_result: PredictionResult, now_time_as_string_for_file_name: str) -> str:
        model_params_as_string_for_file_name = prediction_result.get_model_params_as_string_for_file_name()
        sequence_variation = prediction_result.sequence_variation
        file_name = prediction_result.file_name
        seq_record_name = prediction_result.seq_record_name
        seq_record_key = '{}.{}.{}.{}'.format(file_name, seq_record_name, model_params_as_string_for_file_name, now_time_as_string_for_file_name)
        seq_name = sequence_variation.get_title()
        seq_key = '{}.{}'.format(seq_record_key, seq_name)
        bed_file_name_seq = '{}.bed'.format(seq_key)

        return bed_file_name_seq
        
    def format(self, prediction_result: PredictionResult) -> Iterable[str]:
        sequence_variation = prediction_result.sequence_variation
        model_params_as_string = prediction_result.get_model_params_as_string()
        seq_name = sequence_variation.get_title()
        minimum_sequence_length = prediction_result.minimum_sequence_length
        seq_len = prediction_result.seq_len
        labels = np.asarray(prediction_result.labeled)
        confidence = np.asarray(prediction_result.confidence_scores)
        assert labels.shape == confidence.shape
        mask = labels > 0
        if not mask.any():
            return
        mask_int = mask.astype(np.int8)
        diff = np.diff(mask_int)
        starts = np.where(diff == 1)[0] + 1
        ends = np.where(diff == -1)[0] + 1
        if mask[0]:
            starts = np.r_[0, starts]
        if mask[-1]:
            ends = np.r_[ends, labels.shape[0]]

        label_id = 1
        for start_idx, end_idx in zip(starts, ends):
            candidate_length = end_idx - start_idx

            if candidate_length <= minimum_sequence_length:
                continue

            candidate_indices = np.arange(start_idx, end_idx, dtype=np.int64)
            candidate_start, candidate_end = sequence_variation.derive_candidate_start_and_end(
                seq_len,
                candidate_indices,
            )

            avg_conf = float(np.mean(confidence[candidate_start:candidate_end]))

            bed_name = '{},{},{}'.format(model_params_as_string, seq_name,label_id)

            yield f'0\t{candidate_start}\t{candidate_end}\t{bed_name}\t{avg_conf}'

            label_id += 1