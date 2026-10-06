# Invoice Editor

The editor runs locally in your browser, with Python providing OCR and visual-region detection. Uploaded documents are processed on this computer and are not sent to a remote service.

## Run on Windows

1. Open PowerShell in this folder.
2. Install the Python dependencies:

   ```powershell
   python -m pip install -r requirements.txt
   ```

3. Start the app:

   ```powershell
   python app.py
   ```

4. Open <http://127.0.0.1:8765> in your browser.

The first OCR scan may take longer while the OCR engine initializes. Open images or PDFs with **Open image**; PDFs are rendered from their first page. The detector returns OCR text boxes and visual proposals for buttons, images, and other graphics. Select any detected region to edit it, or draw a region manually when a scan, unusual font, or layout is not recognized.

Stop the local server with `Ctrl+C`.
