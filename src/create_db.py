from tqdm import tqdm
import torch
import gc
import os
import numpy as np
from sklearn.metrics import cohen_kappa_score
from model import TripletLoss, TripletNet, Identity
from dataset import QueryExtractor, EmbeddingDataset
from torchvision import transforms
import torchvision.models as models
from torch.utils.data import DataLoader



def create_embeddings_db(model_weights_path, device, img_dir="./data/oxbuild/images/", fts_dir="./fts/"):
    
    # Create transformss
    transforms_test = transforms.Compose([transforms.Resize(460),
                                        transforms.FiveCrop(448),                                 
                                        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                                        ])

    # Create query img list
    query_img_list = [os.path.join(img_dir, file) for file in os.listdir(img_dir)]

    # Create dataset
    eval_dataset = EmbeddingDataset(img_dir, query_img_list, transforms=transforms_test)
    eval_loader = DataLoader(eval_dataset, batch_size=1, num_workers=0, shuffle=False)

    # Create embedding network
    resnet_model = models.resnet101(pretrained=False)
    model = TripletNet(resnet_model)
    model.load_state_dict(torch.load(model_weights_path))
    model.to(device)
    model.eval()

    # Create features
    with torch.no_grad():
        for idx, image in enumerate(tqdm(eval_loader)):
            # Move image to device and get crops
            image = image.to(device)
            bs, ncrops, c, h, w = image.size()

            # Get output
            output = model.get_embedding(image.view(-1, c, h, w))
            output = output.view(bs, ncrops, -1).mean(1).cpu().numpy()

            # Save fts
            save_path = os.path.join(fts_dir, query_img_list[idx].replace(".jpg", ""))
            np.save(save_path, output)
        

if __name__ == '__main__':
    # Create cuda parameters
    use_cuda = torch.cuda.is_available()
    np.random.seed(2019)
    torch.manual_seed(2019)
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Available device = ", device)

    create_embeddings_db("./weights/temp-triplet_model.pth", device)