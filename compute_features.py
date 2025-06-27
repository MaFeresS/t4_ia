import torch
import clip
from torchvision import transforms, models
import numpy as np
from PIL import Image
import os

# data_dir = '/hd_data/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/'
# image_dir = os.path.join(data_dir, 'JPEGImages')
# val_file = 'data/voc_val.txt'
# data_dir = '/hd_data/Paris/'
# image_dir = os.path.join(data_dir, 'paris')
# val_file = 'data/val_paris.txt'

#MODEL = 'resnet34'
#MODEL = 'dinov2'
MODEL = 'clip'

data_dir = 'VocPascal/'
image_dir = os.path.join(data_dir, 'JPEGImages/JPEGImages')
list_of_images = os.path.join(data_dir, 'val_voc.txt')
if __name__ == '__main__':
    #reading data
    with open(list_of_images, "r+") as file: 
        files = [f.split('\t') for f in file]
        
    # check GPU 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # defining the image preprocessing
    if MODEL != 'clip':
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]),
            ])
    #load de model     
    model = None 
    if MODEL == 'resnet34' :
        model = models.resnet34(pretrained=True).to(device)
        model.fc = torch.nn.Identity() 
    #you can add more models
    if MODEL == 'clip':
        model, preprocess = clip.load("ViT-B/32", device=device)
        model = model.encode_image
    if MODEL == 'dinov2':
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(device)

    dim = 512
    if MODEL=="dinov2":
        dim=384
    
    #Pasamos la imagen por el modelo
    with torch.no_grad():        
        n_images = len(files)
        features = np.zeros((n_images, dim), dtype = np.float32)        
        for i, file in enumerate(files) :                
            corrected_file=file[0]
            filename = os.path.join(image_dir, corrected_file)
            image = Image.open(filename+".jpg").convert('RGB')
            image = preprocess(image).unsqueeze(0).to(device)
            features[i,:] = model(image).cpu()[0,:]
            if i%100 == 0 :
                print(f'{i}/{n_images}')            
                
        feat_file = os.path.join(f'feat_{MODEL}_VocPascal.npy')
        np.save(feat_file, features)
        print('saving data ok')