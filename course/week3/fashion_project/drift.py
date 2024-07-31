# in python
import torch
from torchvision import transforms

prod_data = torch.load('./data/production/dataset.pt')
prod_images = prod_data['images']
prod_image = prod_images[0]  # vary to see a few

# Save an image to disk
prod_image = transforms.ToPILImage()(prod_image)
prod_image.save('./test0.png') # take a look at this