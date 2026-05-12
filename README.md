# Doc Intel Agent

An agentic AI system that autonomously processes financial documents.

## What it does
- Reads digital and handwritten/scanned PDFs
- Classifies document type automatically  
- Extracts all relevant fields using LLM
- Validates extracted data
- Checks financial consistency (totals vs line items)
- Flags handwritten documents for human review

## Tech Stack
Python, LangChain, LangGraph, PyMuPDF, TrOCR, OpenCV, OpenRouter

## Document Types Supported
- Invoices
- Bank Statements
- Forms
- Handwritten/Scanned documents
