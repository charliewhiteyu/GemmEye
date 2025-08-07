# GemmEye 🌱👁️

![GemmEye Logo](GemmEyeLogo.png)

## Overview

GemmEye is a Python application that leverages Google's Gemma-3n vision-language model to analyze cadastral maps and determine whether specific map sections are connected or disconnected. This tool is designed for quickly assessing land parcel connectivity.

## Features

- 🗺️ **Cadastral Map Analysis**: Automatically analyze cadastral maps to determine connectivity between land parcels
- 🤖 **AI-Powered**: Uses Google's state-of-the-art Gemma-3n vision-language model
- 📄 **PDF Support**: Direct processing of PDF cadastral maps
- 🖼️ **Image Processing**: Robust image handling and preprocessing
- ⚡ **Fast Inference**: Optimized for quick analysis and results
- 🔧 **Configurable**: Easy to customize for different map types and questions

## Installation

1. **Clone the repository**:
   ```bash
   git clone <your-repository-url>
   cd gemmeye
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the Gemma-3n model**:
   The application expects the Gemma-3n model to be available in your Kaggle Hub cache. Make sure you have the model downloaded to:
   ```
   ~/.cache/kagglehub/models/google/gemma-3n/transformers/gemma-3n-e2b-it/2
   ```

## Usage

### Basic Usage

```python
from GemmEye import GemmEyeAnalyzer, ImageProcessor

# Initialize the analyzer
analyzer = GemmEyeAnalyzer()

# Convert PDF to image (if needed)
png_path = ImageProcessor.convert_pdf_to_image("./Data/Connected_Map.pdf")

# Load the image
image = ImageProcessor.load_image_from_path(png_path)

# Analyze the cadastral map
prompt = "Are the two boxed cadastral maps, 34 and 34-1, connected or disconnected?"
inference_time = analyzer.analyze_cadastral_map(prompt, image)
```

### Running the Example

To run the included example:

```bash
python GemmEye.py
```

This will analyze the sample cadastral maps in the `Data/` directory.

## Project Structure

```
gemmeye/
├── GemmEye.py              # Main application code
├── GemmEyeLogo.png         # Project logo
├── README.md               # This file
├── requirements.txt        # Python dependencies
└── Data/                   # Sample cadastral maps
    ├── Connected_Map.pdf   # Example of connected parcels
    └── Disconnected_Map.pdf # Example of disconnected parcels
```

## Configuration

### Model Configuration

You can customize the model path by providing it during initialization:

```python
analyzer = GemmEyeAnalyzer(model_path="/path/to/your/model")
```

### Analysis Parameters

The `analyze_cadastral_map` method accepts several parameters:

- `prompt` (str): The question about the cadastral map
- `image` (PIL.Image): The cadastral map image to analyze
- `max_new_tokens` (int): Maximum number of tokens to generate (default: 32)
- `transpose_image` (bool): Whether to transpose the image tensor (default: True)

## Sample Data

The `Data/` directory contains sample cadastral maps:

- **Connected_Map.pdf**: Example showing connected land parcels (34 and 34-1)
- **Disconnected_Map.pdf**: Example showing disconnected land parcels (31-1 and 34-1)

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- PIL (Pillow)
- pdf2image
- numpy
- See `requirements.txt` for complete list

## System Requirements

- **Memory**: At least 8GB RAM recommended
- **Storage**: ~5GB for model files
- **CPU**: Multi-core processor recommended for faster inference

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Google for the Gemma-3n vision-language model
- The Transformers library by Hugging Face
- The cadastral mapping community for inspiration

## Support

If you encounter any issues or have questions, please open an issue on the GitHub repository.

---

**Note**: This tool is designed for analysis purposes. Always verify results with official cadastral records and consult with qualified surveyors for legal or official determinations.

