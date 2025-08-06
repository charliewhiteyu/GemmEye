"""
GemmEye - Cadastral Map Analysis Tool

A Python application that uses Google's Gemma-3n model to analyze cadastral maps
and determine whether specific map sections are connected or disconnected.

Author: Your Name
Version: 1.0.0
"""

import os
import psutil
from time import time
from typing import Tuple, Optional
import numpy as np
from PIL import Image as PILImage
from pdf2image import convert_from_path

import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, TextStreamer

import gc
import ctypes


class GemmEyeAnalyzer:
    """
    A class for analyzing cadastral maps using Google's Gemma-3n vision-language model.
    
    This class handles model loading, image processing, and inference for determining
    whether cadastral map sections are connected or disconnected.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the GemmEye analyzer with the specified model.
        
        Args:
            model_path (str, optional): Path to the Gemma model. If None, uses default cache path.
        """
        self.model_path = model_path or self._get_default_model_path()
        self.processor = None
        self.model = None
        self._load_model()
    
    def _get_default_model_path(self) -> str:
        """
        Get the default model path from the Kaggle Hub cache.
        
        Returns:
            str: Default path to the Gemma-3n model
        """
        return os.path.expanduser(
            "~/.cache/kagglehub/models/google/gemma-3n/transformers/gemma-3n-e2b-it/2"
        )
    
    def _load_model(self) -> None:
        """
        Load the Gemma-3n processor and model for inference.
        
        Raises:
            FileNotFoundError: If the model path doesn't exist
            RuntimeError: If model loading fails
        """
        try:
            print(f"Loading model from: {self.model_path}")
            
            # Load the processor
            self.processor = AutoProcessor.from_pretrained(self.model_path)
            
            # Load the model with optimized settings
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_path,
                torch_dtype="auto",  # Automatically choose the best dtype
                device_map="cpu",    # Use CPU for inference
            ).eval()
            
            print("Model loaded successfully!")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}")
    
    def _create_system_prompt(self) -> str:
        """
        Create the system prompt for the cadastral map analysis task.
        
        Returns:
            str: The system prompt for the AI model
        """
        return """
        You are a smart AI expert in answering cadastral map questions.
        Just answer to the point, do not elaborate.
        Your job is to answer whether two cadastral maps are connected or disconnected.
        """
    
    def analyze_cadastral_map(
        self, 
        prompt: str, 
        image: PILImage.Image, 
        max_new_tokens: int = 32, 
        transpose_image: bool = True
    ) -> float:
        """
        Analyze a cadastral map image and answer the given prompt.
        
        Args:
            prompt (str): The question about the cadastral map
            image (PIL.Image): The cadastral map image to analyze
            max_new_tokens (int): Maximum number of tokens to generate
            transpose_image (bool): Whether to transpose the image tensor
            
        Returns:
            float: Time taken for inference in seconds
        """
        start_time = time()
        
        # Prepare the conversation messages
        system_prompt = self._create_system_prompt()
        messages = [{
            "role": "system",
            "content": [
                {"type": "text", "text": system_prompt}
            ],
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt}
            ]
        }]
        
        # Process the inputs
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(self.model.device, dtype=self.model.dtype)
        
        # Transpose image if requested (some models require this)
        if transpose_image:
            inputs["pixel_values"] = inputs["pixel_values"].permute(0, 1, 3, 2)
        
        # Get input length for reference
        input_len = inputs["input_ids"].shape[-1]
        
        # Generate response with streaming output
        _ = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=1.0,
            top_p=0.95,
            top_k=64,
            streamer=TextStreamer(self.processor.tokenizer, skip_prompt=True)
        )
        
        # Calculate and return inference time
        total_time = round(time() - start_time, 2)
        return total_time


class ImageProcessor:
    """
    Utility class for handling image processing operations.
    """
    
    @staticmethod
    def load_image_from_path(img_path: str) -> PILImage.Image:
        """
        Load and convert an image to RGB format.
        
        Args:
            img_path (str): Path to the image file
            
        Returns:
            PIL.Image: The loaded image in RGB format
            
        Raises:
            FileNotFoundError: If the image file doesn't exist
            IOError: If the image cannot be opened
        """
        try:
            image = PILImage.open(img_path).convert("RGB")
            return image
        except FileNotFoundError:
            raise FileNotFoundError(f"Image file not found: {img_path}")
        except Exception as e:
            raise IOError(f"Failed to open image {img_path}: {str(e)}")
    
    @staticmethod
    def convert_pdf_to_image(pdf_path: str, dpi: int = 300) -> str:
        """
        Convert a PDF file to PNG image format.
        
        Args:
            pdf_path (str): Path to the PDF file
            dpi (int): Resolution for the conversion
            
        Returns:
            str: Path to the generated PNG file
            
        Raises:
            FileNotFoundError: If the PDF file doesn't exist
            RuntimeError: If PDF conversion fails
        """
        try:
            if not os.path.exists(pdf_path):
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
            png_path = pdf_path.replace(".pdf", ".png")
            
            # Convert PDF to images
            png_images = convert_from_path(pdf_path, dpi=dpi)
            
            # Save the first page as PNG
            if png_images:
                png_images[0].save(png_path, "PNG")
                print(f"PDF converted to PNG: {png_path}")
                return png_path
            else:
                raise RuntimeError("No pages found in PDF")
                
        except Exception as e:
            raise RuntimeError(f"Failed to convert PDF to image: {str(e)}")


def main():
    """
    Main function to demonstrate the GemmEye cadastral map analysis.
    """
    # Configuration
    DATA_DIR = "./Data"
    
    # Example configurations for different map analyses
    configs = [
        {
            "pdf_path": os.path.join(DATA_DIR, "Connected_Map.pdf"),
            "prompt": "Are the two boxed cadastral maps, 34 and 34-1, connected or disconnected?"
        },
        {
            "pdf_path": os.path.join(DATA_DIR, "Disconnected_Map.pdf"),
            "prompt": "Are the two boxed cadastral maps, 31-1 and 34-1, connected or disconnected?"
        }
    ]
    
    # Initialize the analyzer
    try:
        analyzer = GemmEyeAnalyzer()
        
        # Process the disconnected map example (current default)
        config = configs[1]  # Using disconnected map
        
        print(f"Processing: {config['pdf_path']}")
        print(f"Question: {config['prompt']}")
        
        # Convert PDF to image
        png_path = ImageProcessor.convert_pdf_to_image(config["pdf_path"])
        
        # Load the image
        image = ImageProcessor.load_image_from_path(png_path)
        
        # Analyze the cadastral map
        inference_time = analyzer.analyze_cadastral_map(
            prompt=config["prompt"],
            image=image,
            max_new_tokens=32,
            transpose_image=False
        )
        
        print(f"\nInference completed in {inference_time} seconds")
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")


if __name__ == "__main__":
    main()

