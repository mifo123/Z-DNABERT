from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import PlainTextResponse, JSONResponse
import tempfile
import pathlib
import traceback
from main import run_zdnabert_analysis

app = FastAPI()


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    print("Internal Server Error:")
    traceback.print_exc()
    return JSONResponse(status_code=500, content={"detail": str(exc)})


@app.post("/analyse", response_class=PlainTextResponse)
async def analyse(
    model: str = Form(...),
    confidence_threshold: float = Form(0.5),
    min_seq_length: int = Form(10),
    check_reverse_complement: bool = Form(False),
    use_cuda: bool = Form(False),
    fasta_file: UploadFile = File(...)
):
    print(f"Received analysis request with parameters:")
    print(f"  model = {model}")
    print(f"  confidence_threshold = {confidence_threshold}")
    print(f"  min_seq_length = {min_seq_length}")
    print(f"  check_reverse_complement = {check_reverse_complement}")
    print(f"  use_cuda = {use_cuda}")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".fasta") as tmp_fasta:
        try:
            contents = await fasta_file.read()
            tmp_fasta.write(contents)
            tmp_fasta_path = pathlib.Path(tmp_fasta.name)

        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to read uploaded file: {e}")

    try:
        bed_lines = run_zdnabert_analysis(
            model=model,
            input_fasta=tmp_fasta_path,
            confidence_threshold=confidence_threshold,
            min_seq_length=min_seq_length,
            check_reverse_complement=check_reverse_complement,
            use_cuda=use_cuda
        )

        if not bed_lines:
            return "No predictions found."

        print("\n".join(bed_lines))
        return "\n".join(bed_lines)

    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        print("TRACEBACK:")
        print(traceback.format_exc())
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        print("TRACEBACK:")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")

    finally:
        tmp_fasta_path.unlink(missing_ok=True)
