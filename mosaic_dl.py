from pathlib import Path
from multiprocessing import Pool
import argparse
import numpy as np
import torch
#from sklearn.decomposition import PCA
import umap
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader

# backends priority from least to greatest

device = "cpu"

if torch.backends.mps.is_available():
    device = "mps"

if torch.cuda.is_available():
    device = "cuda"

print(f"using device: {device}")

import clip

class FlatDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform
        
    def __getitem__(self, index):
        image_path = self.image_paths[index]
        x = Image.open(image_path)
        if self.transform is not None:
            x = self.transform(x)
        return x
    
    def __len__(self):
        return len(self.image_paths)



def create_mosaic(all_img_paths, dim, pre_crop, img_size, out_name):
    model, preprocess = clip.load('ViT-B/32')
    model = model.to(device)
    embeddings = torch.zeros(len(all_img_paths), 512, device=device)
    print("Generating clip embeddings")
    chunk_size = 16
    dataset = FlatDataset(all_img_paths, preprocess)
    loader = DataLoader(dataset, batch_size=chunk_size, shuffle=False, num_workers=4)

    #chunks = list(divide_chunks(list(enumerate(all_img_paths)), chunk_size))
    #print(f"Made {len(chunks)} chunks of size {chunk_size}")

    with torch.no_grad():

        for idx, batch in enumerate(tqdm(loader)):
            batch_embeds = model.encode_image(batch.to(device))
            start = idx * chunk_size
            end = start + len(batch)
            embeddings[start:end] = batch_embeds

        #for idx, p in tqdm(chunks):
        #    img = preprocess().unsqueeze(0).to(device)
        #    embeddings[idx] = model.encode_image(img).squeeze()

    X = embeddings.cpu().numpy()
    #pca = PCA(n_components=2)
    #pca_vals = pca.fit_transform(X)
    #print(f"Explained variance by first 2 principal components: {pca.explained_variance_ratio_.sum()}")
    print("Running umap")
    reduced_dim_dat = umap.UMAP(low_memory=True, verbose=True).fit_transform(X)   
    comps_with_idx = [{"img_idx": idx, "comps": comps } for idx, comps in enumerate(reduced_dim_dat)]
    print("Sorting grid")
    # sort x 
    x_sorted = sorted(comps_with_idx, key=lambda comps: comps["comps"][0])
    grid_sorted = []
    for i in range(dim):
        row = x_sorted[i*dim:(i+1)*dim]
        row_sorted = sorted(row, key=lambda comps: comps["comps"][1])
        grid_sorted.append(row_sorted)
    crop_amt = (pre_crop - img_size) / 2
    total_dim = dim*img_size
    main_img = np.zeros((total_dim,total_dim,4), dtype=np.uint8)
    print("Creating Tiles")
    for y_idx, row in tqdm(list(enumerate(grid_sorted))):
        for x_idx, comps in enumerate(row):
            img_idx = comps["img_idx"]
            img = Image.open(all_img_paths[img_idx])
            resized_img = img.resize((pre_crop, pre_crop))
            if (resized_img.mode != "RGBA"):
                # replace pixels matching alpha_val with transparency
                new_img = np.ones((pre_crop,pre_crop,4), dtype=np.uint8)*255
                new_img[:,:,:3] = resized_img
                # from https://github.com/PWhiddy/PokemonRedExperiments/blob/master/MapWalkingVis.ipynb
                alpha_val = np.array([255, 255,  255, 255], dtype=np.uint8)
                alpha_mask = (new_img == alpha_val).all(axis=2).reshape(pre_crop,pre_crop,1)
                resized_img = Image.fromarray( np.where(alpha_mask, np.array([[[0,0,0,0]]]), new_img).astype(np.uint8) )
            cropped_img = resized_img.crop((crop_amt,crop_amt,pre_crop-crop_amt, pre_crop-crop_amt))
            main_img[
                x_idx*img_size:(x_idx+1)*img_size, 
                y_idx*img_size:(y_idx+1)*img_size] = np.asarray(cropped_img)
    im = Image.fromarray(main_img)
    im.save(out_name)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("img_dir")
  parser.add_argument("grid_dim", type=int)
  parser.add_argument("out_name", default="mosaic_out.png")
  parser.add_argument("img_size", type=int, default=256)
  parser.add_argument("precrop_size", type=int, default=512)
  args = parser.parse_args()
  pth = Path(args.img_dir)
  print("Finding all images")
  all_paths = list(pth.glob("*.png")) + list(pth.glob("*.jpeg")) + list(pth.glob("*.jpg"))
  print(f"{len(all_paths)} images found")
  create_mosaic(all_paths, args.grid_dim, args.precrop_size, args.img_size, args.out_name)
