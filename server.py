import traceback
import tempfile
import pathlib
import asyncio
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import PlainTextResponse

from main import run_zdnabert_analysis

app = FastAPI()

@app.post("/analyse", response_class=PlainTextResponse)
async def analyse(
    model: str = Form(...),
    confidence_threshold: float = Form(0.5),
    min_seq_length: int = Form(10),
    check_reverse_complement: bool = Form(False),
    use_cuda: bool = Form(False),
    fasta_file: UploadFile = File(...)
):
    tmp_fasta_path = None
    try:
        contents = await fasta_file.read()

        if not contents.strip():
            raise HTTPException(status_code=400, detail="Uploaded FASTA file is empty.")

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
            use_cuda=use_cuda
        )

        if not bed_lines:
            return "No predictions found."

        return "\n".join(bed_lines)

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
