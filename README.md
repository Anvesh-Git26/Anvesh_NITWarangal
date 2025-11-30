# 🚀 Neuro-Symbolic Invoice Extractor (BFHL Datathon)

## 📌 Overview
This solution implements a **Neuro-Symbolic Architecture** to extract structured data from complex medical invoices with mathematical precision. It combines state-of-the-art **Multimodal AI (Gemini 2.0 Flash)** for visual extraction with a deterministic **Python Logic Engine** to verify, audit, and self-correct the data before submission.

## ⚡ Key Differentiators
1.  **Forensic Fraud Detection:** Implements **Error Level Analysis (ELA)** to detect digital tampering (e.g., whitener, font injection) by analyzing JPEG compression artifacts.
2.  **Self-Healing Mathematical Core:** Automatically corrects OCR hallucinations by enforcing the constraint `Qty × Rate = Amount`. If the AI misreads a number, the math engine fixes it.
3.  **Advanced Logic Trap Handling:**
    * **Column Swaps:** Automatically detects and fixes swapped "Rate" and "Amount" columns (common in clinic bills).
    * **Ghost Header Filtering:** intelligently filters out category headers (e.g., "Room Charges") to prevent double-counting of line items.
    * **Lump Sum Normalization:** Handles items with missing unit prices by inferring `Qty=1`.
4.  **Financial Taxonomy:** Classifies rows into `ITEM`, `TAX`, `DISCOUNT`, and `ADJUSTMENT` to ensure the final reconciled total is accurate ($Items + Tax - Discount$).

## 🛠️ Tech Stack
* **Framework:** FastAPI (Python 3.13)
* **AI Model:** Google Gemini 2.0 Flash (with robust fallback to 2.5/Pro)
* **Forensics:** OpenCV (Headless) with Numpy
* **Deployment:** Dockerized container on Render (Cloud)

## 🔗 Live API Endpoint
**Method:** `POST`
**URL:** `https://anvesh-nitwarangal.onrender.com/extract-bill-data`

### **Request Format**
```json
{
  "document": "[https://example.com/invoice_image.jpg](https://example.com/invoice_image.jpg)"
}
