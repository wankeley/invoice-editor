"""Local Python server and OCR/visual-region detector for the invoice editor."""

from __future__ import annotations

import base64
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageOps


ROOT = Path(__file__).resolve().parent
MAX_REQUEST_BYTES = 50 * 1024 * 1024
MAX_PIXELS = 50_000_000
_ocr_engine = None
_ocr_lock = threading.Lock()


def get_ocr_engine():
    global _ocr_engine
    if _ocr_engine is None:
        with _ocr_lock:
            if _ocr_engine is None:
                from rapidocr import RapidOCR

                _ocr_engine = RapidOCR()
    return _ocr_engine


def decode_image(data: bytes) -> Image.Image:
    Image.MAX_IMAGE_PIXELS = MAX_PIXELS
    try:
        image = Image.open(BytesIO(data))
        if image.width * image.height > MAX_PIXELS:
            raise ValueError("Images must be 50 megapixels or smaller.")
        image = ImageOps.exif_transpose(image).convert("RGB")
        image.load()
    except (OSError, Image.DecompressionBombError) as exc:
        raise ValueError("The uploaded file is not a readable image.") from exc
    if image.width * image.height > MAX_PIXELS:
        raise ValueError("Images must be 50 megapixels or smaller.")
    return image


def image_from_data_url(value: str) -> Image.Image:
    match = re.fullmatch(r"data:image/(?:png|jpeg|webp);base64,([A-Za-z0-9+/=]+)", value)
    if not match:
        raise ValueError("Expected a base64-encoded PNG, JPEG, or WebP image.")
    try:
        return decode_image(base64.b64decode(match.group(1), validate=True))
    except (ValueError, base64.binascii.Error) as exc:
        raise ValueError("The image data could not be decoded.") from exc


def _iou(a: dict, b: dict) -> float:
    x0, y0 = max(a["x0"], b["x0"]), max(a["y0"], b["y0"])
    x1, y1 = min(a["x1"], b["x1"]), min(a["y1"], b["y1"])
    intersection = max(0, x1 - x0) * max(0, y1 - y0)
    area_a = (a["x1"] - a["x0"]) * (a["y1"] - a["y0"])
    area_b = (b["x1"] - b["x0"]) * (b["y1"] - b["y0"])
    union = area_a + area_b - intersection
    return intersection / union if union else 0


def _normalize_ocr_result(result, width: int, height: int) -> list[dict]:
    """Accept both RapidOCR's legacy row-list and newer result-object formats."""
    rows = []
    if isinstance(result, tuple):
        result = result[0]
    if hasattr(result, "boxes") and hasattr(result, "txts"):
        scores = getattr(result, "scores", [])
        rows = [
            (box, text, scores[index] if index < len(scores) else 0.0)
            for index, (box, text) in enumerate(zip(result.boxes, result.txts))
        ]
    elif result:
        rows = result

    regions = []
    for row in rows:
        if not isinstance(row, (list, tuple)) or len(row) < 2:
            continue
        polygon, text = row[0], str(row[1]).strip()
        score = row[2] if len(row) > 2 else 0.0
        if not text:
            continue
        try:
            points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
            x0 = max(0, int(np.floor(points[:, 0].min())))
            y0 = max(0, int(np.floor(points[:, 1].min())))
            x1 = min(width, int(np.ceil(points[:, 0].max())))
            y1 = min(height, int(np.ceil(points[:, 1].max())))
            confidence = float(score)
        except (TypeError, ValueError, IndexError):
            continue
        if x1 <= x0 or y1 <= y0:
            continue
        regions.append({
            "x0": x0, "y0": y0, "x1": x1, "y1": y1,
            "type": "text", "text": text,
            "confidence": max(0.0, min(1.0, confidence)),
        })
    return regions


def _visual_regions(rgb: np.ndarray, text_regions: list[dict]) -> list[dict]:
    """Find larger bordered/color-block regions likely to be buttons or images."""
    height, width = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    scale = max(1.0, max(width, height) / 1800)
    kernel_size = max(3, int(round(5 * scale)) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    candidates = []

    # Closing connects table-cell and control outlines while retaining large blocks.
    block_size = min(31, (min(width, height) // 2) * 2 - 1)
    if block_size < 3:
        return []
    for thresholded in (
        cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, block_size, 7,
        ),
        cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, block_size, 7,
        ),
    ):
        connected = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, box_width, box_height = cv2.boundingRect(contour)
            area = box_width * box_height
            if area < max(350, width * height * 0.0003):
                continue
            if area > width * height * 0.65 or box_width < 24 or box_height < 14:
                continue
            ratio = box_width / box_height
            if ratio < 0.35 or ratio > 18:
                continue
            fill_ratio = cv2.contourArea(contour) / max(1, area)
            if fill_ratio < 0.32:
                continue
            region = {
                "x0": x, "y0": y, "x1": x + box_width, "y1": y + box_height,
                "type": "graphic", "text": "", "confidence": 0.0,
            }
            if any(_iou(region, text) > 0.28 for text in text_regions):
                continue
            crop = rgb[y:y + box_height, x:x + box_width]
            color_std = float(np.std(crop))
            if area > width * height * 0.012 and color_std > 22:
                region["type"] = "image"
            elif 1.1 <= ratio <= 12 and area < width * height * 0.18:
                region["type"] = "button"
            candidates.append(region)

    # Prefer the tighter proposal when the two threshold polarities find the same block.
    candidates.sort(key=lambda box: (
        box["type"] == "graphic",
        (box["x1"] - box["x0"]) * (box["y1"] - box["y0"]),
    ))
    kept = []
    for candidate in candidates:
        if not any(_iou(candidate, existing) > 0.68 for existing in kept):
            kept.append(candidate)
    return kept


def detect_regions(image: Image.Image) -> dict:
    rgb = np.asarray(image)
    height, width = rgb.shape[:2]
    median_gray = float(np.median(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)))
    polarity = 1 if median_gray < 128 else -1
    threshold = max(60, min(220, median_gray + 130)) if polarity > 0 else max(35, min(195, median_gray - 100))
    result = get_ocr_engine()(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    text_regions = _normalize_ocr_result(result, width, height)
    visual_regions = _visual_regions(rgb, text_regions)
    regions = sorted(
        text_regions + visual_regions,
        key=lambda box: (box["y0"], box["x0"], box["type"] != "text"),
    )
    counts = {
        kind: sum(region["type"] == kind for region in regions)
        for kind in ("text", "button", "image", "graphic")
    }
    return {
        "width": width, "height": height, "regions": regions, "counts": counts,
        "polarity": polarity, "threshold": threshold,
    }


class EditorHandler(BaseHTTPRequestHandler):
    server_version = "InvoiceEditor/1.0"

    def do_GET(self):
        if self.path not in ("/", "/TextSwap.html"):
            self.send_error(404, "Not found")
            return
        try:
            content = (ROOT / "TextSwap.html").read_bytes()
        except OSError:
            self.send_error(500, "TextSwap.html could not be read")
            return
        self._send(200, content, "text/html; charset=utf-8")

    def do_POST(self):
        if self.path not in ("/api/detect", "/api/open"):
            self.send_error(404, "Not found")
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > MAX_REQUEST_BYTES:
                raise ValueError("Upload must be between 1 byte and 50 MB.")
            body = self.rfile.read(length)
            if self.path == "/api/detect":
                payload = json.loads(body)
                image = image_from_data_url(payload.get("image", ""))
                response = detect_regions(image)
            else:
                response = self._open_upload(body)
            self._send(200, json.dumps(response).encode("utf-8"), "application/json")
        except ImportError:
            self._json_error(503, "A required Python package is missing. Run: python -m pip install -r requirements.txt")
        except (ValueError, json.JSONDecodeError) as exc:
            self._json_error(400, str(exc))
        except Exception as exc:
            self.log_error("Request failed: %s", exc)
            self._json_error(500, "Processing failed. Check the server console for details.")

    def _open_upload(self, body: bytes) -> dict:
        filename = self.headers.get("X-File-Name", "document")
        if filename.lower().endswith(".pdf") or body.startswith(b"%PDF"):
            try:
                import pymupdf
            except ImportError as exc:
                raise ImportError("Install PyMuPDF to open PDF files.") from exc
            document = pymupdf.open(stream=body, filetype="pdf")
            if not document.page_count:
                raise ValueError("The PDF has no pages.")
            page = document.load_page(0)
            scale = min(2, (MAX_PIXELS / max(1, page.rect.width * page.rect.height)) ** 0.5)
            pixmap = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False)
            image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
            image.load()
            notice = f"Opened page 1 of {document.page_count}."
        else:
            image = decode_image(body)
            notice = ""
        output = BytesIO()
        image.save(output, format="PNG")
        encoded = base64.b64encode(output.getvalue()).decode("ascii")
        return {
            "image": f"data:image/png;base64,{encoded}",
            "width": image.width,
            "height": image.height,
            "notice": notice,
        }

    def _json_error(self, status: int, message: str):
        self._send(status, json.dumps({"error": message}).encode("utf-8"), "application/json")

    def _send(self, status: int, content: bytes, content_type: str):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(content)

    def log_message(self, format_string: str, *args):
        print(f"[{self.log_date_time_string()}] {format_string % args}")


def main():
    host, port = "127.0.0.1", 8765
    server = ThreadingHTTPServer((host, port), EditorHandler)
    print(f"Invoice Editor running at http://{host}:{port}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping Invoice Editor.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
