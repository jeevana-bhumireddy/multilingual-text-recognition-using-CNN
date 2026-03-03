# OCR Text Scanner

A real-time document scanner that captures images via webcam and extracts text using OCR. The tool applies a multi-stage image preprocessing pipeline to maximise text recognition accuracy before passing the image to Tesseract.

---

## How It Works

1. **Capture** — Opens a live webcam feed. Press `s` to snap a photo or `q` to quit.
2. **Preprocess** — The captured image is put through a pipeline to clean and enhance it:
   - Rotated to correct orientation
   - Colors inverted for contrast enhancement
   - Converted to grayscale
   - Binarized using Otsu's thresholding
   - Noise removed via morphological transformations (dilation, erosion, closing, median blur)
   - Borders detected and cropped out using contour detection
3. **Extract** — The cleaned image is passed to Tesseract OCR, which outputs the detected text to the console.

---

## Requirements

- Python 3.x
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) installed at:
  ```
  C:\Program Files\Tesseract-OCR\tesseract.exe
  ```
- A connected webcam

### Python Dependencies

```bash
pip install opencv-python numpy matplotlib pytesseract
```

---

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/ocr-text-scanner.git
   cd ocr-text-scanner
   ```

2. Install dependencies:
   ```bash
   pip install opencv-python numpy matplotlib pytesseract
   ```

3. Install Tesseract OCR from [here](https://github.com/tesseract-ocr/tesseract/releases) and ensure it is installed at the path above (or update the path in the script).

4. Create the output directory:
   ```bash
   mkdir Outputs
   ```

---

## Usage

```bash
python scanner.py
```

| Key | Action                          |
|-----|---------------------------------|
| `s` | Capture image and extract text  |
| `q` | Quit without capturing          |

The scanned image is saved to `./Outputs/ScannedImage.jpg` and the extracted text is printed to the console.

---

## Preprocessing Pipeline

| Step                  | Technique                              | Purpose                            |
|-----------------------|----------------------------------------|------------------------------------|
| Rotation              | `cv2.ROTATE_90_CLOCKWISE`             | Correct document orientation       |
| Color inversion       | `cv2.bitwise_not`                     | Improve contrast on dark documents |
| Grayscale conversion  | `cv2.COLOR_BGR2GRAY`                  | Reduce to single channel           |
| Binarization          | Otsu's thresholding                   | Separate text from background      |
| Noise removal         | Dilation → Erosion → Morph Close      | Remove small artifacts             |
| Smoothing             | Median blur (kernel size 3)           | Reduce remaining noise             |
| Border removal        | Contour detection + bounding rect crop| Focus OCR on text content only     |

---

## Tech Stack

- **OpenCV** — Image capture, transformation, and preprocessing
- **NumPy** — Kernel and array operations
- **Tesseract OCR + pytesseract** — Text extraction
- **Matplotlib** — Available for visualisation and debugging

---

## Project Structure

```
ocr-text-scanner/
├── scanner.py        # Main script
├── Outputs/          # Saved captured images
└── README.md
```

---

## Known Limitations

- Tesseract path is currently hardcoded for Windows. Update the path in `scanner.py` for macOS/Linux:
  ```python
  # macOS / Linux
  pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'
  ```
- OCR accuracy depends on lighting conditions and document quality at capture time.
- Currently supports English (`lang='eng'`) only.
