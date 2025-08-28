# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Any

# Third Party
from fastapi import APIRouter
from starlette.requests import Request
from starlette.responses import PlainTextResponse

router = APIRouter()


@router.post("/run_script")
async def run_script(request: Request):
    form_data = await request.form()
    script_file = form_data.get("script")

    if not script_file or not hasattr(script_file, "file"):
        return PlainTextResponse("No script file provided", status_code=400)

    script_content = await script_file.read()

    try:
        restricted_globals = {
            "__builtins__": {
                "print": print,
                "str": str,
                "int": int,
                "float": float,
                "list": list,
                "dict": dict,
                "tuple": tuple,
                "set": set,
            },
            "app": request.app,
        }
        restricted_locals: dict[str, Any] = {}

        exec(script_content, restricted_globals, restricted_locals)

        result = restricted_locals.get("result", "Script executed successfully")
        return PlainTextResponse(str(result), media_type="text/plain")

    except Exception as e:
        return PlainTextResponse(f"Error executing script: {str(e)}", status_code=500)
