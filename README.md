# SenPa-MAE
This is the official code for the manuscript: [SenPa-MAE: Sensor Parameter Aware Masked Autoencoder for Multi-Satellite Self-Supervised Pretraining](https://arxiv.org/pdf/2408.11000). The paper presents **SenPa-MAE**, a transformer architecture that encodes sensor parameters from multispectral signals into image embeddings, enabling cross-sensor training and represent a step towards sensor-independent inference. The model uses a sensor parameter encoding module and a data augmentation strategy to allow pre-training on imagery from satellites with varying sensor specifications, enhancing its adaptability to different Earth observation missions.

In this work we make use of data of the satellites `Sentinel-2` (S2), `Landsat` (LS) and `Superdove` (SD) from PlanetLabs. Generally this method is not generally restricted to those sensors and can easily be expanded.



## Data

Parts of the satellite data for pre-training is unfortunately not granted with a license that would allow sharing it here. Still under `./data` the ids for all images used for pre-training of the sensors S2, LS and SD are given. Generally the exact data distribution should not be to critical as long as it covers all land-cover types in a reasonable manner.

```json
{
    "ROI_0": {
        "Sentinel2": [
            "S2A_MSIL2A_20230913T083601_N0509_R064_T37UDB_20230913T130804.SAFE"
        ],
        "SuperDove": [
            "20230913_074044_65_24b0_3B_AnalyticMS_SR_8b.tif",
            "20230913_082218_40_2416_3B_AnalyticMS_SR_8b.tif",
            "20230913_074615_27_24ca_3B_AnalyticMS_SR_8b.tif",
            "20230913_074035_45_24b0_3B_AnalyticMS_SR_8b.tif",
            "20230913_082220_51_2416_3B_AnalyticMS_SR_8b.tif",
            "20230913_074612_97_24ca_3B_AnalyticMS_SR_8b.tif",
            "20230913_074617_57_24ca_3B_AnalyticMS_SR_8b.tif",
            "20230913_074619_86_24ca_3B_AnalyticMS_SR_8b.tif",
            "20230913_082216_29_2416_3B_AnalyticMS_SR_8b.tif",
            "20230913_074040_05_24b0_3B_AnalyticMS_SR_8b.tif",
            "20230913_074553_57_2464_3B_AnalyticMS_SR_8b.tif"
        ],
        "Landsat": [
            "LC08_L2SP_178021_20230914_20230919_02_T1"
        ]
    },
    "ROI_1": {
        "Sentinel2": [
            "S2B_MSIL2A_20240204T155449_N0510_R054_T18TWL_20240204T200249.SAFE"
        ],
        "SuperDove": [
            "20240204_145925_91_24b5_3B_AnalyticMS_SR_8b.tif",
            "20240204_154555_09_2479_3B_AnalyticMS_SR_8b.tif",
            "20240204_154557_06_2479_3B_AnalyticMS_SR_8b.tif",
            "20240204_154559_02_2479_3B_AnalyticMS_SR_8b.tif",
            ... ]
    ...},
... } 
```

We opt for constant 5m pixel spacing along all data no matter the original sensor resolution. Data is saved in patches of 288x288 pixels and stored in 3 separate folders indicating the sensor name with filenames indicating the image resolution and  pixel spacing. Instead of  performing the downsampling operation on the fly in the data-loader (computational expensive operation) we save all different resolutions to the drive which leads to much faster model training.

```
├───patches
│   ├───Sentinel2
│   │   └───ROI1
│   │   |    └───sample_1_10mGSD_5mPxSp.tif
│   │   |    └───sample_1_15mGSD_5mPxSp.tif
│   │   |    └───sample_1_20mGSD_5mPxSp.tif
│   │   |    └───sample_1_30mGSD_5mPxSp.tif
│   │   |    └───sample_2_10mGSD_5mPxSp.tif
│   │   |    └───sample_2_15mGSD_5mPxSp.tif
│   │   |    └───sample_2_20mGSD_5mPxSp.tif
│   │   |    └───sample_2_30mGSD_5mPxSp.tif
			...
│   ├───Superdove
│   │   └───ROI1
│   │   |    └───sample_1_5mGSD_5mPxSp.tif
│   │   |    └───sample_1_10mGSD_5mPxSp.tif
│   │   |    └───sample_1_15mGSD_5mPxSp.tif
│   │   |    └───sample_1_20mGSD_5mPxSp.tif
│   │   |    └───sample_1_30mGSD_5mPxSp.tif
│   │   |    └───sample_2_5mGSD_5mPxSp.tif
│   │   |    └───sample_2_10mGSD_5mPxSp.tif
│   │   |    └───sample_2_15mGSD_5mPxSp.tif
│   │   |    └───sample_2_20mGSD_5mPxSp.tif
│   │   |    └───sample_2_30mGSD_5mPxSp.tif
			...
```

Modifications of this structure are ofc possible but require according modifications in the here provided data-loader.



## Model

The main model is a modified vision transformer where the main contribution is given by the sensor parameter encoding module. Self-supervised pre-training is conducted with the mask autoencoder strategy. The model parameter can be taken from the config files:

```yaml
model:
  _target_: "model.SenPaMAE"
  encoder:
    _target_: "model.Encoder"
    image_size: 144
    num_channels: 4
    patch_size: 16
    emb_dim: 768
    num_layer: 12
    num_head: 12
    sensor_parameter_embedding_active: True
    channels_seperate_tokens: True
    positional_embedding_3D: False
    maskingfunction: 
      _target_: "maskingfunction.PatchShuffle"
      maskingStategy: "random"
      masking_ratio: 66
      image_size: ${model.encoder.image_size}
      num_channels: ${model.encoder.num_channels}
      patch_size: ${model.encoder.patch_size}
  decoder:
    _target_: "model.Decoder"
    image_size: ${model.encoder.image_size}
    num_channels: ${model.encoder.num_channels}
    patch_size: ${model.encoder.patch_size}
    emb_dim: ${model.encoder.emb_dim}
    channels_seperate_tokens: ${model.encoder.channels_seperate_tokens}
    positional_embedding_3D: ${model.encoder.positional_embedding_3D}
    num_layer: 3
    sensor_parameter_embedding_active: ${model.encoder.sensor_parameter_embedding_active}
    num_head: ${model.encoder.num_head}
```



Most impotently an inactive `sensor parameter encoding module` either in the encoder or decoder part must be replaced by 3d positional embeddings (compare configs or manuscript). 

The responsefunctions used in the `sensor parameter encoding module` can be found under `./responsefunctions` and are taken from [here](https://support.planet.com/hc/en-us/articles/360014290293-Do-you-provide-Relative-Spectral-Response-Curves-RSRs-for-your-satellites), [here](https://sentinels.copernicus.eu/web/sentinel/document-library/latest-documents/-/asset_publisher/EgUy8pfXboLO/content/sentinel-2a-spectral-responses;jsessionid=6F22D73101A6E4ABB84D95FF35A40A42.jvm1?redirect=https%3A%2F%2Fsentinels.copernicus.eu%2Fweb%2Fsentinel%2Fdocument-library%2Flatest-documents%3Bjsessionid%3D6F22D73101A6E4ABB84D95FF35A40A42.jvm1%3Fp_p_id%3D101_INSTANCE_EgUy8pfXboLO%26p_p_lifecycle%3D0%26p_p_state%3Dnormal%26p_p_mode%3Dview%26p_p_col_id%3Dcolumn-1%26p_p_col_pos%3D1%26p_p_col_count%3D2) and [here](https://landsat.gsfc.nasa.gov/satellites/landsat-8/spacecraft-instruments/operational-land-imager/spectral-response-of-the-operational-land-imager-in-band-band-average-relative-spectral-response/).



### Start a training

Start and connect to the docker container:

```bash
cd docker/
bash start_container.sh
```

and run

```python
python main.py --config-name="basemae"
```

for the baseline, or 

```python
python main.py --config-name="senpamae_singleParameterEmbedding"
```

```python
python main.py --config-name="senpamae_doubleParameterEmbedding"
```

for the `SenPaMAE` model.

## Checkpoints
Please contact me (jonathanprexl@gmail.com) if you are interested in model weights.


