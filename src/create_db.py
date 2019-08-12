from tqdm import tqdm
import torch
import gc
import os
import numpy as np
from sklearn.metrics import cohen_kappa_score
from model import TripletNet, create_embedding_net
from dataset import QueryExtractor, EmbeddingDataset
from torchvision import transforms
import torchvision.models as models
from torch.utils.data import DataLoader
from utils import perform_pca_on_single_vector


def create_embeddings_db_pca(model_weights_path, img_dir, fts_dir):
    """
    Given a model weights path, this function creates a triplet network, loads the parameters and generates the dimension
    reduced (using pca) vectors and save it in the provided feature directory.
    
    Args:
        model_weights_path  : path of trained weights
        img_dir     : directory that holds the images
        fts_dir     : directory to store the embeddings
    
    Returns:
        None
    
    Eg run:
        create_embeddings_db_pca("./weights/oxbuild-exp-3.pth", img_dir="./data/oxbuild/images/", fts_dir="./fts_pca/oxbuild/")
    """
    # Create cuda parameters
    use_cuda = torch.cuda.is_available()
    np.random.seed(2019)
    torch.manual_seed(2019)
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Available device = ", device)

    # Create transforms
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transforms_test = transforms.Compose([transforms.Resize(460),
                                        transforms.FiveCrop(448),                                 
                                        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                                        transforms.Lambda(lambda crops: torch.stack([transforms.Normalize(mean=mean, std=std)(crop) for crop in crops])),
                                        ])

    # Creat image database
    if "paris" in img_dir:
        print("> Blacklisted images must be removed")
        blacklist = ["paris_louvre_000136.jpg",
        "paris_louvre_000146.jpg",
        "paris_moulinrouge_000422.jpg",
        "paris_museedorsay_001059.jpg",
        "paris_notredame_000188.jpg",
        "paris_pantheon_000284.jpg",
        "paris_pantheon_000960.jpg",
        "paris_pantheon_000974.jpg",
        "paris_pompidou_000195.jpg",
        "paris_pompidou_000196.jpg",
        "paris_pompidou_000201.jpg",
        "paris_pompidou_000467.jpg",
        "paris_pompidou_000640.jpg",
        "paris_sacrecoeur_000299.jpg",
        "paris_sacrecoeur_000330.jpg",
        "paris_sacrecoeur_000353.jpg",
        "paris_triomphe_000662.jpg",
        "paris_triomphe_000833.jpg",
        "paris_triomphe_000863.jpg",
        "paris_triomphe_000867.jpg",]
        
        files = os.listdir(img_dir)
        for blacklisted_file in blacklist:
            files.remove(blacklisted_file)

        QUERY_IMAGES = [os.path.join(img_dir, file) for file in sorted(files)]

    else:
        QUERY_IMAGES = [os.path.join(img_dir, file) for file in sorted(os.listdir(img_dir))]

    # Create dataset
    eval_dataset = EmbeddingDataset(img_dir, QUERY_IMAGES, transforms=transforms_test)
    eval_loader = DataLoader(eval_dataset, batch_size=1, num_workers=0, shuffle=False)

    # Create embedding network
    resnet_model = create_embedding_net()
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

            # Perform pca
            output = perform_pca_on_single_vector(output)

            # Save fts
            img_name = (QUERY_IMAGES[idx].split("/")[-1]).replace(".jpg", "")
            save_path = os.path.join(fts_dir, img_name)
            np.save(save_path, output.flatten())

            del output, image
            gc.collect()


# if __name__ == '__main__':
#     create_embeddings_db_pca("./weights/oxbuild-exp-3.pth", img_dir="./data/oxbuild/images/", fts_dir="./fts_pca/oxbuild/")