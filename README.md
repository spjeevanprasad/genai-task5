# Neural Style Transfer

## Description
This project applies Neural Style Transfer using a pre-trained VGG19 model.

It combines:
- Content image structure
- Style image textures

to generate a stylized output image.

## Files
- genai-task5.py : Main Python script
- content.jpg : Content image
- style.jpg : Style image
- output.jpg : Generated output

## How to Run

### Install dependencies
pip install -r requirements.txt

## Output
The final stylized image is saved as:
output.jpg

## Working
- Load content and style images
- Extract features using VGG19
- Compute content loss and style loss
- Optimize a generated image
- Save final stylized output

## Author
Jeevan
