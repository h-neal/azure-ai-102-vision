from dotenv import load_dotenv
import os
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt

# Import Azure AI Vision client libraries
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential

def main():
    """
    Main function to load environment variables, authenticate the Azure client,
    and handle user input for choosing an image to analyze.
    """
    global cv_client

    try:
        # Load Azure credentials from .env file
        load_dotenv()
        ai_endpoint = os.getenv('AI_SERVICE_ENDPOINT')
        ai_key = os.getenv('AI_SERVICE_KEY')

        # Initialize the Azure AI Vision client
        cv_client = ImageAnalysisClient(
            endpoint=ai_endpoint,
            credential=AzureKeyCredential(ai_key)
        )

        # Provide a simple CLI menu for selecting an image
        print('\n1: Use Read API for image (Lincoln.jpg)\n2: Read handwriting (Note.jpg)\nAny other key to quit\n')
        command = input('Enter a number:')
        
        # Based on user input, call the function to process the selected image
        if command == '1':
            image_file = os.path.join('images', 'Lincoln.jpg')
            GetTextRead(image_file)
        elif command == '2':
            image_file = os.path.join('images', 'Note.jpg')
            GetTextRead(image_file)

    except Exception as ex:
        # Handle and display any errors that occur
        print("Error:", ex)

def GetTextRead(image_file):
    """
    Reads text from an image using Azure AI Vision Read API,
    overlays bounding boxes on the text, and saves the result.
    """
    print('\nReading text from image...\n')

    # Load the image in binary format
    with open(image_file, "rb") as f:
        image_data = f.read()

    # Call Azure AI Vision to analyze the image for text using the Read visual feature
    result = cv_client.analyze(
        image_data=image_data,
        visual_features=[VisualFeatures.READ]
    )

    # Check that text was actually detected in the image
    if result.read is not None and result.read.blocks:
        print("\nExtracted Text:")

        # Load the image using PIL for drawing overlays
        image = Image.open(image_file)

        # Prepare a figure for saving the annotated image
        fig = plt.figure(figsize=(image.width / 100, image.height / 100))
        plt.axis('off')  # Hide axes
        draw = ImageDraw.Draw(image)
        color = 'cyan'  # Color used for bounding boxes

        # Iterate through blocks of text
        for block in result.read.blocks:
            for line in block.lines:
                # Print the detected line text
                print(f"    {line.text}")

                # Draw bounding polygon around the line
                line_polygon = [(pt.x, pt.y) for pt in line.bounding_polygon]
                draw.polygon(line_polygon, outline=color, width=3)
                print(" Bounding Polygon:", line_polygon)

                # Draw bounding boxes for each word in the line
                for word in line.words:
                    word_polygon = [(pt.x, pt.y) for pt in word.bounding_polygon]
                    print(f"    Word: '{word.text}', Bound Polygon: {word_polygon}, Confidence: {word.confidence:.4f}")
                    draw.polygon(word_polygon, outline=color, width=3)

        # Display and save the image with overlays
        plt.imshow(image)
        plt.tight_layout(pad=0)
        outputfile = 'text.jpg'
        fig.savefig(outputfile)
        print('\nResults saved in', outputfile)

# Entry point to run the program
if __name__ == "__main__":
    main()