# 🚀 Neuro-Symbolic Invoice Extractor (BFHL Datathon)

## 📌 Overview
This solution implements a **Neuro-Symbolic Architecture** to extract structured data from medical invoices with 100% mathematical consistency. It combines **Gemini 1.5 Flash (Vision)** for extraction with a deterministic **Python Logic Engine** to verify and self-correct values.

## ⚡ Key Differentiators
1.  **Forensic Fraud Detection:** Uses **Error Level Analysis (ELA)** to detect digital tampering (whitener/font edits) on the invoice.
2.  **Self-Healing Math:** Automatically corrects OCR hallucinations by enforcing `Qty × Rate = Amount`.
3.  **Logic Trap Handling:**
    * **Column Swaps:** Detects and fixes swapped Rate/Amount columns.
    * **Ghost Headers:** Filters out category headers to prevent double-counting.
    * **Lump Sums:** Normalizes items with missing unit prices.

## 🛠️ Tech Stack
* **Framework:** FastAPI (Python)
* **AI Model:** Gemini 1.5 Flash
* **Forensics:** OpenCV (Headless)
* **Hosting:** Render (Cloud)

## 🔗 API Endpoint
**Method:** `POST`
**URL:** `https://anvesh-nitwarangal.onrender.com/extract-bill-data`

**Request Body:**
```json
{
  "document": "IMAGE_URL_HERE"
}
