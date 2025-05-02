from dotenv import load_dotenv
import os
import sys
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
from azure.cognitiveservices.vision.computervision import ComputerVisionClient  # Correct import
from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes
from msrest.authentication import CognitiveServicesCredentials

def main():
    try:
        # Load environment variables
        load_dotenv()
        ai_endpoint = os.getenv('AI_SERVICE_ENDPOINT')
        ai_key = os.getenv('AI_SERVICE_KEY')

        if not ai_endpoint or not ai_key:
            raise ValueError("Please set AI_SERVICE_ENDPOINT and AI_SERVICE_KEY in your .env file")

        # Authenticate client
        cv_client = ComputerVisionClient(ai_endpoint, CognitiveServicesCredentials(ai_key))

        # Image path
        image_file = 'images/street.jpg'
        if len(sys.argv) > 1:
            image_file = sys.argv[1]

        # Open image file
        with open(image_file, "rb") as image_stream:
            # Analyze image
            features = [
                VisualFeatureTypes.description,
                VisualFeatureTypes.tags,
                VisualFeatureTypes.objects
            ]
            analysis = cv_client.analyze_image_in_stream(image_stream, visual_features=features)

        # Print results
        print("\nAnalysis results for:", image_file)

        # Caption
        if analysis.description and analysis.description.captions:
            print("\nCaption:")
            for caption in analysis.description.captions:
                print(f" '{caption.text}' (confidence: {caption.confidence:.2f})")

        # Tags
        if analysis.tags:
            print("\nTags:")
            for tag in analysis.tags:
                print(f" {tag.name} ({tag.confidence:.2f})")

        # Objects
        if analysis.objects:
            print("\nObjects:")
            for obj in analysis.objects:
                print(f" {obj.object_property} ({obj.confidence:.2f})")

    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    main()
