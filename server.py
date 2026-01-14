import traceback
import tempfile
import pathlib
import asyncio
from datetime import datetime
import hashlib

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import PlainTextResponse

from main import run_zdnabert_analysis

app = FastAPI()

RESULTS_DIR = pathlib.Path("/zdnabert2_results")
CACHE_VERSION = "v1"


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


@app.post("/analyse", response_class=PlainTextResponse)
async def analyse(
    model: str = Form(...),
    confidence_threshold: float = Form(0.5),
    min_seq_length: int = Form(10),
    check_reverse_complement: bool = Form(False),
    use_cuda: bool = Form(False),
    fasta_file: UploadFile = File(...),
):
    tmp_fasta_path = None
    try:
        contents = await fasta_file.read()

        if not contents.strip():
            raise HTTPException(status_code=400, detail="Uploaded FASTA file is empty.")

        # cache key + cesta k výslednému súboru v zdieľanom volume
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

        # cache hit – súbor už existuje, nič nepočítame
        if out_path.exists():
            return file_name  # vraciame len názov súboru, nie cestu

        # cache miss – treba spustiť analýzu
        with tempfile.NamedTemporaryFile(delete=False, suffix=".fasta") as tmp_fasta:
            tmp_fasta.write(contents)
            tmp_fasta_path = pathlib.Path(tmp_fasta.name)

        bed_lines = await asyncio.to_thread(
            run_zdnabert_analysis,
            model=model,
            input_fasta=tmp_fasta_path,
            confidence_threshold=confidence_threshold,
            min_seq_length=min_seq_length,
            check_reverse_complement=check_reverse_complement,
            use_cuda=False,  # zatiaľ ignorujeme use_cuda z formulára
        )

        if not bed_lines:
            # žiadne predikcie – neexistuje výsledný súbor, vrátime prázdny string
            return ""

        # zapíš výsledok do zdieľaného volume
        with out_path.open("w") as f:
            for line in bed_lines:
                f.write(line)
                if not line.endswith("\n"):
                    f.write("\n")

        # vraciame len názov súboru
        return file_name

    except FileNotFoundError as e:
        print(f"[FileNotFoundError] {e}")
        raise HTTPException(status_code=400, detail=str(e))

    except ValueError as e:
        print("TRACEBACK:")
        traceback.print_exc()
        raise HTTPException(status_code=422, detail=str(e))

    except Exception as e:
        print("TRACEBACK:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")

    finally:
        if tmp_fasta_path and tmp_fasta_path.exists():
            tmp_fasta_path.unlink(missing_ok=True)
