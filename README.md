# sp-mosaic

Script to create a mosiac of many images, visually sorted by the first 2 principal components of their CLIP embedding.

Usage:
```python
from pathlib import Path
import mosaic 

path = Path("your/dir/with/imgs")
all_paths = path.glob("*.png")           
mosaic.create_mosaic(all_paths, 25, 512, 256, "test_grid.png")
```
