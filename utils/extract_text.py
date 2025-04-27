import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import logging
import fitz  # PyMuPDF
from PIL import Image
import pytesseract
from utils.logger import logger, log_token_usage  # Importing logger and log_token_usage function

def extract_text_from_pdf(pdf_path):
    logger.info(f"Starting to extract text from PDF: {pdf_path}")
    text = ""
    try:
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text += page.get_text()
        logger.info(f"Successfully extracted text from PDF: {pdf_path}")
        
        # Estimate token usage based on text length
        tokens = len(text.split())
        log_token_usage(
            model="pdf-extraction",  # Custom model name for PDF text extraction
            prompt_tokens=tokens,
            completion_tokens=0,  # No completion tokens
            total_tokens=tokens
        )
    except Exception as e:
        logger.error(f"Error extracting text from PDF {pdf_path}: {e}")
    return text

def extract_text_from_image(image_path):
    logger.info(f"Starting to extract text from image: {image_path}")
    text = ""
    try:
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image)
        logger.info(f"Successfully extracted text from image: {image_path}")
        
        # Estimate token usage based on text length
        tokens = len(text.split())
        log_token_usage(
            model="image-extraction",  # Custom model name for image text extraction
            prompt_tokens=tokens,
            completion_tokens=0,  # No completion tokens
            total_tokens=tokens
        )
    except Exception as e:
        logger.error(f"Error extracting text from image {image_path}: {e}")
    return text

def extract_text(file_path):
    logger.info(f"Starting text extraction from file: {file_path}")
    ext = os.path.splitext(file_path)[-1].lower()
    try:
        if ext == ".pdf":
            return extract_text_from_pdf(file_path)
        elif ext in [".png", ".jpg", ".jpeg"]:
            return extract_text_from_image(file_path)
        else:
            raise ValueError("Unsupported file format")
    except Exception as e:
        logger.error(f"Error extracting text from file {file_path}: {e}")
        return ""
