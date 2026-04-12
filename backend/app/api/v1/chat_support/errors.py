from __future__ import annotations

import httpx


def exception_detail(exc: Exception) -> str:
    if isinstance(exc, httpx.HTTPStatusError):
        detail = str(exc).strip()
        response_text = exc.response.text.strip()
        if response_text:
            return f"{detail}: {response_text}"
        if detail:
            return detail
    detail = str(exc).strip()
    if detail:
        return detail
    cause = exc.__cause__ or exc.__context__
    if cause is not None:
        cause_detail = str(cause).strip() or repr(cause)
        if cause_detail:
            return f"{exc.__class__.__name__}: {cause_detail}"
    repr_detail = repr(exc)
    if repr_detail:
        return repr_detail
    return exc.__class__.__name__
