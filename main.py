import os
import json
import tempfile
import mimetypes
from typing import Any, Dict, List, Optional

import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import google.generativeai as genai

# Optional: import your fraud engine if present
try:
    from fraud_engine import detect_fraud_advanced
except ImportError:
    detect_fraud_advanced = None


# -----------------------------
# FastAPI app setup
# -----------------------------
app = FastAPI(title="BFHL Neuro-Symbolic Invoice Extractor")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------
# Pydantic models
# -----------------------------
class InvoiceRequest(BaseModel):
    """
    'document' can be:
    - a plain URL string
    - a dict containing a URL
    - a nested structure with url/type/url inside
    """
    document: Any


# -----------------------------
# Helper: Extract URL from arbitrary "url type doc"
# -----------------------------
def extract_url_from_document(document: Any) -> str:
    """
    Make the API resilient to different 'url type doc' formats.

    Supported examples:
    - "https://.../file.pdf"
    - {"url": "..."}
    - {"type": "url", "url": "..."}
    - {"type": "url", "value": "..."}
    - {"doc": {"url": "..."}}
    - [ {"url": "..."}, ... ]
    """
    # Case 1: plain string
    if isinstance(document, str):
        doc = document.strip()
        if doc.startswith("http://") or doc.startswith("https://"):
            return doc

    # Case 2: dict – try common patterns
    if isinstance(document, dict):
        # Direct common keys
        for key in ("url", "link", "href", "document_url"):
            val = document.get(key)
            if isinstance(val, str) and (val.startswith("http://") or val.startswith("https://")):
                return val

        # {"type": "url", "value": "..."} kind of structure
        doc_type = str(document.get("type", "")).lower()
        if doc_type == "url":
            for k in ("value", "data", "doc_value", "document"):
                val = document.get(k)
                if isinstance(val, str) and (val.startswith("http://") or val.startswith("https://")):
                    return val

        # Fallback: recursively search nested dicts
        for v in document.values():
            if isinstance(v, (dict, list, str)):
                try:
                    return extract_url_from_document(v)
                except HTTPException:
                    continue

    # Case 3: list – search elements
    if isinstance(document, list):
        for item in document:
            try:
                return extract_url_from_document(item)
            except HTTPException:
                continue

    raise HTTPException(
        status_code=400,
        detail="Unsupported 'document' format. Could not find a valid URL.",
    )


# -----------------------------
# Helper: Download remote file
# -----------------------------
def download_document_to_tempfile(doc_url: str) -> str:
    """
    Downloads the document from the given URL into a NamedTemporaryFile,
    choosing the extension based on Content-Type (pdf/jpg/png/etc.).
    Returns the local file path.
    """

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": (
            "text/html,application/xhtml+xml,application/xml;"
            "q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8"
        ),
        "Accept-Language": "en-US,en;q=0.9",
        "Connection": "keep-alive",
    }

    try:
        resp = requests.get(doc_url, headers=headers, stream=True, timeout=60)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error downloading document: {e}")

    if resp.status_code != 200:
        raise HTTPException(status_code=400, detail=f"Download failed: HTTP {resp.status_code}")

    content_type = resp.headers.get("Content-Type", "").split(";")[0].lower()

    # Decide extension
    if content_type == "application/pdf":
        suffix = ".pdf"
    elif content_type in ("image/jpeg", "image/jpg"):
        suffix = ".jpg"
    elif content_type == "image/png":
        suffix = ".png"
    else:
        guessed = mimetypes.guess_extension(content_type)
        suffix = guessed if guessed else ".bin"

    try:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        with tmp as f:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        return tmp.name
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to store downloaded file: {e}")


# -----------------------------
# Logic engine: verify & reconcile line items
# -----------------------------
def safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        if isinstance(x, (int, float)):
            return float(x)
        if isinstance(x, str):
            val = x.replace(",", "").strip()
            if not val:
                return None
            return float(val)
    except Exception:
        return None
    return None


def verify_and_reconcile(pagewise_line_items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Takes the raw pagewise line items from the model and:
    - normalizes quantities/rates/amounts to floats
    - filters obvious headers
    - computes total_item_count and reconciled_amount
    """
    clean_pages: List[Dict[str, Any]] = []
    all_items_flat: List[Dict[str, Any]] = []

    for page in pagewise_line_items or []:
        page_num = page.get("page_number")
        items = page.get("items", [])

        clean_items = []
        for raw_item in items or []:
            name = (raw_item.get("item_name") or "").strip()

            qty = safe_float(raw_item.get("item_quantity"))
            rate = safe_float(raw_item.get("item_rate"))
            amt = safe_float(raw_item.get("item_amount"))

            # Skip completely empty lines
            if not name and qty is None and rate is None and amt is None:
                continue

            # Skip obvious headers: no quantity/rate/amount and all caps-ish
            if (
                not qty
                and not rate
                and not amt
                and len(name) <= 30
                and name.upper() == name
            ):
                continue

            item = {
                "item_name": name,
                "item_quantity": qty if qty is not None else 0.0,
                "item_rate": rate if rate is not None else 0.0,
                "item_amount": amt if amt is not None else 0.0,
            }

            clean_items.append(item)
            flat_item = dict(item)
            flat_item["page_number"] = page_num
            all_items_flat.append(flat_item)

        if clean_items:
            clean_pages.append(
                {
                    "page_number": page_num,
                    "items": clean_items,
                }
            )

    # Reconciled amount = sum of cleaned item_amounts (no double counting)
    reconciled_amount = round(
        sum(i.get("item_amount") or 0.0 for i in all_items_flat), 2
    )
    total_item_count = len(all_items_flat)

    return {
        "pagewise_line_items": clean_pages,
        "total_item_count": total_item_count,
        "reconciled_amount": reconciled_amount,
    }


# -----------------------------
# Gemini setup
# -----------------------------
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise RuntimeError("GOOGLE_API_KEY environment variable is not set")

genai.configure(api_key=GOOGLE_API_KEY)

gemini_model = genai.GenerativeModel(
    "gemini-1.5-flash",
    generation_config={
        "temperature": 0.1,
        "top_p": 0.8,
        "top_k": 32,
        "max_output_tokens": 2048,
        "response_mime_type": "application/json",
    },
)


# -----------------------------
# Root endpoint (health check)
# -----------------------------
@app.get("/")
def root():
    return {"status": "ok", "message": "BFHL Neuro-Symbolic Invoice Extractor running"}


# -----------------------------
# Main extraction endpoint
# -----------------------------
@app.post("/extract-bill-data")
def extract_bill_data(request: InvoiceRequest):
    temp_filename: Optional[str] = None

    try:
        # 1. Normalize document to URL
        doc_url = extract_url_from_document(request.document)

        # 2. Download into a temp file (handles PDF / JPG / PNG)
        temp_filename = download_document_to_tempfile(doc_url)

        # 3. Optional fraud detection
        fraud_data = None
        if detect_fraud_advanced is not None:
            try:
                fraud_data = detect_fraud_advanced(temp_filename)
            except Exception as e:
                # Don't fail the whole call if fraud engine crashes
                fraud_data = {"error": str(e)}

        # 4. Upload file to Gemini
        try:
            file_ref = genai.upload_file(path=temp_filename)
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Failed to upload file to Gemini: {e}"
            )

        # 5. Prompt Gemini to extract structured line items
        prompt = """
You are an expert medical invoice parsing engine.

Task:
- Read this medical bill / invoice from the attached file.
- Extract ONLY the chargeable line items (no headers, no grand total rows, no subtotals).
- For each line item, fill:
  - item_name        : string (the description as text)
  - item_quantity    : number (use 1 if not mentioned)
  - item_rate        : number (unit rate, can be 0 if only amount is present)
  - item_amount      : number (total line amount)

Group items page-wise.

Return STRICTLY JSON with this structure:

{
  "pagewise_line_items": [
    {
      "page_number": 1,
      "items": [
        {
          "item_name": "string",
          "item_quantity": 1.0,
          "item_rate": 100.0,
          "item_amount": 100.0
        }
      ]
    }
  ]
}

Rules:
- Do NOT include subtotal / total / tax summary rows as line items.
- If only a single lump-sum amount is present for a row, use quantity=1 and rate=amount.
- Use page_number starting from 1.
- DO NOT add any extra keys. DO NOT wrap in markdown. Return raw JSON only.
"""

        try:
            gemini_resp = gemini_model.generate_content(
                [file_ref, prompt]
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Gemini generate_content failed: {e}",
            )

        # 6. Parse JSON from Gemini response
        try:
            raw_json = gemini_resp.text
            raw_data = json.loads(raw_json)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to parse JSON from Gemini response: {e}",
            )

        pagewise_line_items = raw_data.get("pagewise_line_items", [])

        # 7. Apply logic engine to verify & reconcile
        logic_output = verify_and_reconcile(pagewise_line_items)

        # 8. Token usage (if available)
        usage = getattr(gemini_resp, "usage_metadata", None)
        token_data = None
        if usage is not None:
            try:
                token_data = {
                    "prompt_tokens": getattr(usage, "prompt_token_count", None),
                    "candidates_tokens": getattr(usage, "candidates_token_count", None),
                    "total_tokens": getattr(usage, "total_token_count", None),
                }
            except Exception:
                token_data = None

        # 9. Final response
        return {
            "is_success": True,
            "token_usage": token_data,
            "data": {
                "pagewise_line_items": logic_output["pagewise_line_items"],
                "total_item_count": logic_output["total_item_count"],
                "reconciled_amount": logic_output["reconciled_amount"],
                "fraud_analysis": fraud_data,
            },
        }

    except HTTPException:
        # Re-raise HTTPExceptions as-is
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")
    finally:
        if temp_filename and os.path.exists(temp_filename):
            try:
                os.remove(temp_filename)
            except Exception:
                pass


# -----------------------------
# Local dev entrypoint
# -----------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=True,
    )
