# Environments
```shell
conda env create -n <env> -f environment.yaml
```

# Run program
```python
python main.py
```

# Outputs
File Name  | Details
------------- | -------------
HDR_image.hdr  | The radiance map
tonemap_photographic_global.JPG  | The tone mapped image with Reinhard's algorithm (global mapping)
tonemap_photographic_local.JPG  | The tone mapped image with Reinhard's algorithm (local mapping)