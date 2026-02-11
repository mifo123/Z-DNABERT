import traceback
import tempfile
import pathlib
import hashlib
import os
import threading
import time

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import PlainTextResponse

from main import run_zdnabert_analysis
from src.status_file import update_status_file

app = FastAPI()

RESULTS_DIR = pathlib.Path("/zdnabert2_results")
CACHE_VERSION = "v1"

# maximálny počet paralelných výpočtov (default 3, dá sa zmeniť env premennou)
MAX_PARALLEL_JOBS = int(os.getenv("ZDNABERT_MAX_PARALLEL_JOBS", "3"))
job_slots = threading.BoundedSemaphore(MAX_PARALLEL_JOBS)


def make_cache_key(
    model: str,
    confidence_threshold: float,
    min_seq_length: int,
    check_reverse_complement: bool,
    fasta_bytes: bytes,
) -> str:
    h = hashlib.sha256()
    h.update(fasta_bytes)
    digest = h.hexdigest()[:16]

    parts = [
        CACHE_VERSION,
        model,
        f"ct={confidence_threshold}",
        f"min={min_seq_length}",
        f"rc={int(bool(check_reverse_complement))}",
        digest,
    ]
    return "__".join(parts)


def fasta_length_from_bytes(contents: bytes) -> int:
    """
    Spočíta počet báz v FASTA (ignoruje hlavičky '>' a prázdne riadky).
    """
    length = 0
    for line in contents.splitlines():
        line = line.strip()
        if not line or line.startswith(b">"):
            continue
        length += len(line)
    return length


def run_and_save_to_file(
    *,
    model: str,
    confidence_threshold: float,
    min_seq_length: int,
    check_reverse_complement: bool,
    fasta_path: pathlib.Path,
    out_path: pathlib.Path,
    status_path: pathlib.Path,
    sequence_length: int | None = None,
) -> None:
    """
    Beží v worker threade:
    - spustí ZDNABERT2 analýzu,
    - zapisuje do out_path.tmp,
    - po dokončení urobí os.replace(tmp, out_path) (atomické),
    - priebežne aktualizuje status súbor,
    - nakoniec vymaže dočasné súbory a uvoľní slot v semafore.
    """
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    start_time = time.time()

    # QUEUED -> RUNNING (serverová časť, reálny progress pôjde z tqdm)
    try:
        update_status_file(
            status_path,
            {
                "version": 1,
                "file": out_path.name,
                "status": "running",
                "stage": "startup",
                "progress": 0.0,
                "eta_seconds": None,
                "sequence_length": sequence_length,
                "created_at": start_time,
                "started_at": start_time,
                "finished_at": None,
                "error": None,
            },
        )
    except Exception:
        traceback.print_exc()

    try:
        bed_lines = run_zdnabert_analysis(
            model=model,
            input_fasta=fasta_path,
            confidence_threshold=confidence_threshold,
            min_seq_length=min_seq_length,
            check_reverse_complement=check_reverse_complement,
            use_cuda=False,
            status_path=status_path,
            sequence_length=sequence_length,
        )

        if not bed_lines:
            # žiadne predikcie – finálny súbor vôbec nevytvárame
            finish_time = time.time()
            try:
                update_status_file(
                    status_path,
                    {
                        "status": "done",
                        "stage": "done_no_predictions",
                        "progress": 1.0,
                        "eta_seconds": 0.0,
                        "finished_at": finish_time,
                        "error": None,
                    },
                )
            except Exception:
                traceback.print_exc()
            return

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        with tmp_path.open("w") as f:
            for line in bed_lines:
                f.write(line)
                if not line.endswith("\n"):
                    f.write("\n")

        # atomicky vytvorí/nahradí finálny súbor
        os.replace(tmp_path, out_path)

        finish_time = time.time()
        try:
            update_status_file(
                status_path,
                {
                    "status": "done",
                    "stage": "done",
                    "progress": 1.0,
                    "eta_seconds": 0.0,
                    "finished_at": finish_time,
                    "error": None,
                },
            )
        except Exception:
            traceback.print_exc()

    except Exception as e:
        print("TRACEBACK (run_and_save_to_file):")
        traceback.print_exc()
        try:
            update_status_file(
                status_path,
                {
                    "status": "error",
                    "stage": "error",
                    "progress": None,
                    "eta_seconds": None,
                    "finished_at": time.time(),
                    "error": str(e),
                },
            )
        except Exception:
            traceback.print_exc()
    finally:
        # cleanup dočasných súborov
        try:
            fasta_path.unlink(missing_ok=True)
        except Exception:
            pass
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
        # uvoľni slot pre ďalší job
        try:
            job_slots.release()
        except ValueError:
            pass


@app.post("/analyse", response_class=PlainTextResponse)
async def analyse(
    background_tasks: BackgroundTasks,
    model: str = Form(...),
    confidence_threshold: float = Form(0.5),
    min_seq_length: int = Form(10),
    check_reverse_complement: bool = Form(False),
    use_cuda: bool = Form(False),
    fasta_file: UploadFile = File(...),
):
    try:
        contents = await fasta_file.read()

        if not contents.strip():
            raise HTTPException(
                status_code=400, detail="Uploaded FASTA file is empty."
            )

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)

        cache_key = make_cache_key(
            model=model,
            confidence_threshold=confidence_threshold,
            min_seq_length=min_seq_length,
            check_reverse_complement=check_reverse_complement,
            fasta_bytes=contents,
        )

        file_name = f"{cache_key}.bed"
        out_path = RESULTS_DIR / file_name
        tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
        status_path = RESULTS_DIR / f"{cache_key}.status"

        seq_len = fasta_length_from_bytes(contents)

        # 1) výsledok už existuje – hotovo
        if out_path.exists():
            now = time.time()
            try:
                update_status_file(
                    status_path,
                    {
                        "version": 1,
                        "file": out_path.name,
                        "status": "cached",
                        "stage": "done",
                        "progress": 1.0,
                        "eta_seconds": 0.0,
                        "sequence_length": seq_len,
                        "created_at": now,
                        "started_at": now,
                        "finished_at": now,
                        "error": None,
                    },
                )
            except Exception:
                traceback.print_exc()

            return file_name

        # 2) analýza už beží (tmp súbor existuje) – nespúšťaj znova, len vráť file_name
        if tmp_path.exists():
            # status súbor môže alebo nemusí existovať – necháme na klienta, aby si ho čítal
            return file_name

        # 3) ani výsledok, ani tmp – nový beh
        tmp_path.touch(exist_ok=True)

        created_at = time.time()
        try:
            update_status_file(
                status_path,
                {
                    "version": 1,
                    "file": out_path.name,
                    "status": "queued",
                    "stage": "queued",
                    "progress": 0.0,
                    "eta_seconds": None,
                    "sequence_length": seq_len,
                    "created_at": created_at,
                    "started_at": None,
                    "finished_at": None,
                    "error": None,
                },
            )
        except Exception:
            traceback.print_exc()

        # blokujúco zober slot (max MAX_PARALLEL_JOBS bežiacich naraz)
        job_slots.acquire()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".fasta") as tmp_fasta:
            tmp_fasta.write(contents)
            tmp_fasta_path = pathlib.Path(tmp_fasta.name)

        # spusti výpočet v samostatnom threade (nie v event loop threade)
        worker = threading.Thread(
            target=run_and_save_to_file,
            kwargs=dict(
                model=model,
                confidence_threshold=confidence_threshold,
                min_seq_length=min_seq_length,
                check_reverse_complement=check_reverse_complement,
                fasta_path=tmp_fasta_path,
                out_path=out_path,
                status_path=status_path,
                sequence_length=seq_len,
            ),
            daemon=True,
        )
        worker.start()

        # HTTP odpoveď je okamžite – iba názov súboru (kompatibilita)
        return file_name

    except FileNotFoundError as e:
        print(f"[FileNotFoundError] {e}")
        raise HTTPException(status_code=400, detail=str(e))

    except ValueError as e:
        print("TRACEBACK (ValueError):")
        traceback.print_exc()
        raise HTTPException(status_code=422, detail=str(e))

    except HTTPException:
        raise

    except Exception as e:
        print("TRACEBACK (Unexpected):")
        traceback.print_exc()
        raise HTTPException(
            status_code=500, detail=f"Unexpected error: {e}"
        )
