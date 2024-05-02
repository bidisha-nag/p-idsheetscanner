from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib import messages
from django.http import JsonResponse, HttpResponseBadRequest, HttpResponseServerError, FileResponse
from django.core.files.storage import FileSystemStorage
from django.http import FileResponse
import tempfile
from django.conf import settings
import os
from PIL import Image
import cv2
import numpy as np
import fitz  # PyMuPDF
import shutil
import tempfile
import logging
import time

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Global variable to store unique matches
unique_matches = []

def index(request):
    if request.user.is_anonymous:
        return render(request, 'login.html')
    
    # Get any messages that were sent during the request
    messages = messages.get_messages(request)
    
    return render(request, "generate_result.html", {'messages': messages})

def loginUser(request):
    if request.method == "POST":
        username = request.POST.get('username')
        password = request.POST.get('password')
        user = authenticate(username=username, password=password)
        if user is not None:
            login(request, user)
            return redirect("/")
        else:
            messages.error(request, 'Invalid username or password.')
            return render(request, 'login.html')
    return render(request, "login.html")

def logoutUser(request):
    logout(request)
    return render(request, 'login.html')

def generate_result(request):
    if request.user.is_anonymous:
        return render(request, 'login.html')
    
    if request.method == 'POST':
        try:
            start_time = time.time()  # Start time of processing

            # Check if PDF and symbol folder are provided in the request
            pdf_file = request.FILES.get('pdf')
            symbol_folder = request.FILES.getlist('symbol_folder')

            if pdf_file and symbol_folder:
                # Save the PDF file to a temporary location on the server
                fs = FileSystemStorage()
                saved_pdf_path = fs.save(pdf_file.name, pdf_file)

                # Save symbol files to a temporary directory
                temp_dir = tempfile.mkdtemp()
                for file in symbol_folder:
                    with open(os.path.join(temp_dir, file.name), 'wb+') as destination:
                        for chunk in file.chunks():
                            destination.write(chunk)

                # Process the PDF and symbol files
                output_image_path, total_matches, unique_matches_count = process_pdf_and_symbols(saved_pdf_path, temp_dir)

                # Calculate processing time
                end_time = time.time()
                processing_time = end_time - start_time

                # Delete the temporary files after processing
                fs.delete(saved_pdf_path)
                shutil.rmtree(temp_dir)

                # Return JSON response with processing results
                response_data = {
                    'output_image_url': output_image_path,
                    'unique_matches_count': unique_matches_count,
                    'processing_time': processing_time
                }
                return JsonResponse(response_data)
            else:
                error_message = 'PDF and symbol files are required.'
                return HttpResponseBadRequest(error_message)
        except Exception as e:
            logger.error(f"Error processing files: {e}")
            error_message = 'Error processing files. Please try again.'
            return HttpResponseServerError(error_message)
    else:
        return render(request, 'generate_result.html')

def process_pdf_and_symbols(pdf_path, symbol_folder_path):
    try:
        # Load the PDF and convert it to an image
        doc = fitz.open(pdf_path)
        page = doc.load_page(0)
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        img.save("temp_image.png")

        # Load the input image
        input_image = cv2.imread('temp_image.png', cv2.IMREAD_COLOR)

        # Check if the input image is loaded successfully
        if input_image is None:
            raise ValueError("Error: Input image could not be loaded.")

        # Check if the input image size is valid
        if input_image.size == 0:
            raise ValueError("Error: Input image size is empty.")

        # Process symbols
        output_image = input_image.copy()
        total_matches = 0
        matched_regions = []

        for filename in os.listdir(symbol_folder_path):
            symbol_path = os.path.join(symbol_folder_path, filename)
            if symbol_path.endswith('.png'):
                matches = process_template_matching(input_image, symbol_path, output_image, matched_regions)
                total_matches += matches

        # Save the final output image
        output_image_path = os.path.join(os.path.dirname(pdf_path), 'output_image.png')
        cv2.imwrite(output_image_path, output_image)

        return output_image_path, total_matches, len(unique_matches)
    except Exception as e:
        logger.error(f"Error processing PDF and symbols: {e}")
        raise

def process_template_matching(input_image, symbol_path, output_image, matched_regions):
    try:
        symbol_img = cv2.imread(symbol_path, cv2.IMREAD_COLOR)

        # Rotate the symbol image at 90, 180, and 270 degrees
        symbol_rotations = [
            symbol_img,
            cv2.rotate(symbol_img, cv2.ROTATE_90_CLOCKWISE),
            cv2.rotate(symbol_img, cv2.ROTATE_180),
            cv2.rotate(symbol_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        ]

        # Convert the input image to grayscale
        input_gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

        # Initialize set to store matched regions
        for symbol_rotated in symbol_rotations:
            symbol_gray = cv2.cvtColor(symbol_rotated, cv2.COLOR_BGR2GRAY)

            # Perform template matching
            for threshold in np.arange(0.74, 0.82, 0.01):
                result = cv2.matchTemplate(input_gray, symbol_gray, cv2.TM_CCOEFF_NORMED)
                loc = np.where(result >= threshold)

                # Store matches
                for pt in zip(*loc[::-1]):
                    if save_unique_match(pt, matched_regions):
                        cv2.rectangle(output_image, pt, (pt[0] +symbol_gray.shape[1], pt[1] + symbol_gray.shape[0]), (0, 255, 0), 2)
                        matched_regions.append(pt)

        return len(matched_regions)
    except Exception as e:
        logger.error(f"Error processing template matching: {e}")
        raise

def save_unique_match(pt, matched_regions, tolerance=15):
    for loc in matched_regions:
        # Check if the new point is within the tolerance radius of an existing point
        if abs(loc[0] - pt[0]) <= tolerance and abs(loc[1] - pt[1]) <= tolerance:
            return False
    # If no nearby point is found, add the new point to the set of matched regions
    matched_regions.append(pt)
    global unique_matches
    unique_matches.append(pt)
    return True
