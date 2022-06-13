from pathlib import Path
import argparse
import numpy as np
import torch
#from sklearn.decomposition import PCA
import umap
from PIL import Image
from tqdm import tqdm
import clip

def create_mosaic(all_img_paths, dim, pre_crop, img_size, out_name):
    model, preprocess = clip.load('ViT-B/32')
    embeddings = torch.zeros(len(all_img_paths), 512)
    print("Generating clip embeddings")
    with torch.no_grad():
        for idx, p in tqdm(list(enumerate(all_img_paths))):
            img = preprocess(Image.open(p)).unsqueeze(0)
            embeddings[idx] = model.encode_image(img).squeeze()
    X = embeddings.numpy()
    #pca = PCA(n_components=2)
    #pca_vals = pca.fit_transform(X)
    #print(f"Explained variance by first 2 principal components: {pca.explained_variance_ratio_.sum()}")
    print("Running umap")
    reduced_dim_dat = umap.UMAP().fit_transform(X)   
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
  all_paths = list(pth.glob("*.png")) + list(pth.glob("*.jpeg"))
  create_mosaic(all_paths, args.grid_dim, args.precrop_size, args.img_size, args.out_name)
