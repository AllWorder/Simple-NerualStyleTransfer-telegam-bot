from PIL import Image, ImageDraw
import torch
import torchvision.transforms as transforms

def image_loader(image_name, image_size=256, device='cpu'):
    """Loads image from directory and returns it as torch.tensor for NN"""
    image = Image.open(image_name)
    loader = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor()])
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

def save_image(tensor, image_name, extension='png'):
    """Saves image as 'image_name.extention' """
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    unloader = transforms.ToPILImage()
    image = unloader(image)
    image = image.save(image_name+'.'+extension)
