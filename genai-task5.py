import torch
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load image
def load_image(image_path, size=400):
    image = Image.open(image_path).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor()
    ])

    image = transform(image).unsqueeze(0)
    return image.to(device)

# Load content & style images
content = load_image("content.jpg")
style = load_image("style.jpg")

# Pretrained VGG19 model
model = models.vgg19(pretrained=True).features.to(device).eval()

# Loss function
mse = torch.nn.MSELoss()

# Gram matrix
def gram_matrix(x):
    _, c, h, w = x.size()
    x = x.view(c, h * w)
    return torch.mm(x, x.t())

# Feature extractor
def get_features(x, model):
    layers = {
        '0': 'conv1',
        '5': 'conv2',
        '10': 'conv3',
        '19': 'conv4',
        '28': 'conv5'
    }

    features = {}
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
    return features

# 🚀 FIX: prevent graph tracking
with torch.no_grad():
    content_features = get_features(content, model)
    style_features = get_features(style, model)

    style_grams = {
        layer: gram_matrix(style_features[layer])
        for layer in style_features
    }

# Generated image (IMPORTANT FIX)
generated = content.clone().detach().requires_grad_(True).to(device)

# Optimizer
optimizer = optim.Adam([generated], lr=0.003)

# Weights
style_weight = 1e6
content_weight = 1

# Training loop
for step in range(300):

    generated_features = get_features(generated, model)

    content_loss = mse(
        generated_features['conv4'],
        content_features['conv4']
    )

    style_loss = 0

    for layer in style_grams:
        gen_gram = gram_matrix(generated_features[layer])
        style_gram = style_grams[layer]
        style_loss += mse(gen_gram, style_gram)

    total_loss = content_weight * content_loss + style_weight * style_loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if step % 50 == 0:
        print(f"Step {step}, Loss: {total_loss.item()}")

# Save output
final_img = generated.squeeze().detach().cpu()
final_img = transforms.ToPILImage()(final_img)
final_img.save("output.jpg")

print("Stylized image saved as output.jpg")