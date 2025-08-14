from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

router = APIRouter()

@router.post("/echo", tags=["tools"], summary="Echo", operation_id="tool_echo_post")
async def echo(req: Request):
    try:
        body = await req.json()
    except Exception:
        body = {}
    return JSONResponse(
        {
            "received": body,
            "now_utc": __import__("datetime").datetime.utcnow().isoformat() + "Z",
        }
    )
