import os
import glob
import numpy as np
import json
import re
import h5py
import warnings
import rasterio
from abc import abstractmethod
import hydra
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

import torch
from torch.utils.data import Dataset

from utils import preprocess_s2

class BaseDataloader(Dataset):

    def __init__(self,
                 topdir_dataset,
                 relative_locations,
                 restrict_train_data,
                 restrict_val_data,
                 train_val_key):

        self.topdir_dataset = topdir_dataset
        self.train_val_key = train_val_key
        
        # load the file with the realive locations of the patches
        relative_locations_file = relative_locations
        with open(relative_locations_file, "r") as f:
            self.relative_locations = json.load(f)[train_val_key]

        # restrict the number of sampels if so specified in the config file
        if train_val_key == "train":
            if restrict_train_data != -1:
                self.relative_locations = self.relative_locations[:restrict_train_data]
        elif train_val_key == "val":
            if restrict_val_data != -1:
                self.relative_locations = self.relative_locations[:restrict_val_data]
        else:
            raise ValueError("Incorect trainvalkey")


        print(f"In total {len(self.relative_locations)} for {train_val_key}")

    @abstractmethod
    def __getitem__(self):
        pass

    def __len__(self):
        return len(self.relative_locations)

class Sentinel2_SuperDove_Landsat(Dataset):
    """
    Sentinel2_SuperDove_Landsat Dataset
    This class loads Sentinel2, SuperDove, and Landsat images with
    different GSD values, pixel spacings, and mixed response functions.
    
    Args:
        topdir_dataset (str): location of the dataset on the drive
        pixelSpacing (str): resampling factor for all satellite images
        patchSize (int): Size of the patch.
        numChannels (int): Number of channels.
        fileformat (str): eighter "hdf5" or "tif" depending on the file format that should be loaded
        downsampleProb (int): Probability of downsampling the images (image augmentation)
        channelMixingProb (int): Probability of mixing the channels (image augmentation)
        channelMixingMaxChannel (int): Maximum number of channels to mix
        locs_responsefunctions (str): Location of the responsefunctions for all sensors
        rois_val (list): List of regions of interest for validation
        rois_test (list): List of regions of interest for testing
        include_s2 (bool): Include Sentinel2 images
        include_sd (bool): Include SuperDove images
        include_ls (bool): Include Landsat images
        train_val_test_key  (str): Key for the train, validation, or test set
    """

    def __init__(self,
                 topdir_dataset,
                 pixelSpacing,
                 patchSize,
                 numChannels,
                 fileformat,
                 downsampleProb,
                 channelMixingProb,
                 channelMixingMaxChannel,
                 locs_responsefunctions,
                 rois_val,
                 rois_test,
                 include_s2,
                 include_sd,
                 include_ls,
                 train_val_test_key):

        super().__init__()

        assert pixelSpacing in ["5mPxSp"], "Not propperly tested for other resolutions"
        self.pixelSpacing = pixelSpacing
        self.topdir_dataset = topdir_dataset
        self.numChannels = numChannels

        assert all([patchSize%x == 0 for x in [5,10,15,20,30]])
        self.patchSize = patchSize
        assert self.patchSize == 720, "Not propperly tested for other resolutions"

        if fileformat == "hdf5":
            self.readFromDisk = self._readFromDisk_h5py
        
        elif fileformat == "tif":
            self.readFromDisk = self._readFromDisk_rasterio
        else:
            raise NotImplementedError

        self.downsampleProb = downsampleProb
        self.channelMixingProb = channelMixingProb
        self.channelMixingMaxChannel = channelMixingMaxChannel
        assert self.channelMixingMaxChannel in [2,3,4]

        rois = set()
        for sensor in ["sentinel2","superdove","landsat"]:
            rois.update(set(os.listdir(os.path.join(topdir_dataset,sensor))))
        rois = list(rois)

        # set the regions for train val and test
        assert train_val_test_key in ["train","val","test"]
        assert all([x in rois for x in rois_val])
        assert all([x in rois for x in rois_test])

        if train_val_test_key == "train":
            for key in rois_val + rois_test:
                assert key in rois, "unknown region of interst to remove"
                rois.remove(key)
        elif train_val_test_key == "val":
            rois = rois_val
        else:
            rois = rois_test

        print(f"Initalize dataloader with mode \"{train_val_test_key}\" with following regions:",rois)
        
        # get locations for sentinel-2 imagary
        s2_locations = []
        for roi in rois:
            assert os.path.isdir(os.path.join(topdir_dataset,"sentinel2"))
            s2_locations += glob.glob(os.path.join(topdir_dataset,"sentinel2",roi,"sentinel2_*_10mGSD_5mPxSp."+fileformat),recursive=True)

        # get all locations for superdove
        sd_locations = []
        for roi in rois:
            assert os.path.isdir(os.path.join(topdir_dataset,"superdove"))
            sd_locations += glob.glob(os.path.join(topdir_dataset,"superdove",roi,"superdove_*_5mGSD_5mPxSp."+fileformat),recursive=True)

        # get all locations for landsat
        ls_locations = []
        for roi in rois:
            assert os.path.isdir(os.path.join(topdir_dataset,"landsat"))
            ls_locations += glob.glob(os.path.join(topdir_dataset,"landsat",roi,"landsat_*_30mGSD_5mPxSp."+fileformat),recursive=True)

        print(f"Found items for sentinel2 // superdove // landsat for mode \"{train_val_test_key}\" -->",len(s2_locations),len(sd_locations),len(ls_locations))

        s2_locations = [("sentinel2",x) for x in s2_locations]
        sd_locations = [("superdove",x) for x in sd_locations]
        ls_locations = [("landsat",x) for x in ls_locations]

        if not include_s2:
            s2_locations = []
        if not include_sd:
            sd_locations = []
        if not include_ls:
            ls_locations = []

        self.locations = s2_locations + sd_locations + ls_locations

        # make sure they are unique
        ids = ["".join(x[1].split("_")[:-2]) for x in self.locations]
        assert len(ids) == len(set(ids)), "ids should be uniuqe"
        
        np.random.shuffle( self.locations )

        # get the response functions
        self.sentinel2_responseFunctions = np.load(os.path.join(locs_responsefunctions,"rfs_sentinel2_a.npy"))
        self.superdove_responseFunctions = np.load(os.path.join(locs_responsefunctions,"rfs_superdove.npy"))
        self.landsat_responseFunctions = np.load(os.path.join(locs_responsefunctions,"rfs_landsat.npy"))

        # channel first
        self.sentinel2_responseFunctions = np.swapaxes(self.sentinel2_responseFunctions,0,1)
        self.superdove_responseFunctions = np.swapaxes(self.superdove_responseFunctions,0,1)
        self.landsat_responseFunctions = np.swapaxes(self.landsat_responseFunctions,0,1)

        return None

    def _ScaleData(self,x):
        x = x.astype("float32")
        return x/10000
    
    def _modifyGSDinPath(self,loc,newGSD):
        pattern = r'(_5mGSD_|_10mGSD_|_30mGSD_)'
        split_list = re.split(pattern, loc)
        return split_list[0] + f"_{newGSD}mGSD_" + split_list[-1]
    
    def _readFromDisk_rasterio(self,location_modified,channelIndex):
        with rasterio.open(location_modified,"r") as src:
            d = src.read(tuple(channelIndex+1))
        return d

    def _readFromDisk_h5py(self,location_modified,channelIndex):
        with h5py.File(location_modified, 'r') as hdf5_file:
            # Access the 'image' dataset directly
            image_dataset = hdf5_file['image']
            # Read data from the dataset
            d = image_dataset[tuple(channelIndex),:,:]
        return d

    def __getitem__(self, i):
    
        sensor, location = self.locations[i]

        if self.pixelSpacing == "5mPxSp":
            if sensor == "sentinel2":
                location = location.replace("_10mGSD_10mPxSp","_10mGSD_5mPxSp")
            elif sensor == "superdove":
                pass
            elif sensor == "landsat":
                location = location.replace("_30mGSD_30mPxSp","_30mGSD_5mPxSp")
            else:
                raise ValueError("unknown sensor")
        else:
            raise ValueError("Problem with differnt resolution batches not implemented yet")

        # we pick a subset of the channels
        # for this first, depending on the sensor,
        # get the potential bands and the corresponing resolutions
        if sensor == "sentinel2":
            potential_bands = list(range(10))
            GSDs = [10,10,10,20,20,20,10,20,20,20]
            responsefunction = self.sentinel2_responseFunctions
        elif sensor == "superdove":
            potential_bands = list(range(8))
            GSDs = [5 for i in range(8)]
            responsefunction = self.superdove_responseFunctions
        elif sensor == "landsat":
            potential_bands = list(range(7))
            GSDs = [30 for i in range(8)]
            responsefunction = self.landsat_responseFunctions
        else:
            raise ValueError("unknown sensor")
        
        # get a random set of channels
        output_composition = []
        for _ in range(self.numChannels):
            
            # if channelMixing is on we might also
            # create syntetic channels by channel superposition
            if np.random.rand() < self.channelMixingProb:
                # get two or more channels to mix
                maxC = np.random.choice(np.arange(2,self.channelMixingMaxChannel+1))
                channelIndex = np.random.choice(potential_bands,maxC,replace=False)
                alphas = np.random.rand(maxC)
                alphas /= alphas.sum() # applitedes for channel mixing
            else:
                channelIndex = np.random.choice(potential_bands,1,replace=False)
                alphas = np.ones(1)

            # depending on the singals there is a minimum GSD
            # if this channel is a composition of two
            # we pick the higher GSD of the two as the minimum
            used_gsds = [GSDs[i] for i in channelIndex]
            min_GSD = max(used_gsds)

            # if we are not at the min resolution 
            # we might do downsampling if active
            if min_GSD < 30:
                if np.random.rand() < self.downsampleProb:
                    potential_GSDs = list( range(min_GSD,31,5) )
                    potential_GSDs.remove(min_GSD)
                    if 25 in potential_GSDs:
                        potential_GSDs.remove(25) # 25m GSD is not in the dataset
                    min_GSD = np.random.choice(potential_GSDs)

            output_composition.append((channelIndex,min_GSD,alphas))

        # load the data and perform channelmixing
        # and downsampling if active
        data = []
        rfs = []
        gsds = []
        locations = []

        for channelIndex, min_GSD, alphas in output_composition:

            channelIndex.sort()

            # load the correct GSD
            location_modified = self._modifyGSDinPath(location,min_GSD)           
            locations.append(location_modified)

            gsds.append(min_GSD)

            d = self.readFromDisk(location_modified,channelIndex)
            d = self._ScaleData(d)
            d = d * alphas[:,None,None]
            d = np.expand_dims( np.mean(d,axis=0), axis=0 )
            data.append(d)
                
            rf = responsefunction[channelIndex]
            rf = rf * alphas[:,None]
            rf = np.expand_dims( np.mean(rf,axis=0), axis=0 )
            rfs.append(rf)

        data = np.concatenate(data,axis=0)

        # random crop
        hw = self.patchSize // 5
        x0,y0 = np.random.randint(0,288-hw,2)
        data = data[:,x0:x0+hw,y0:y0+hw]

        rfs = np.concatenate(rfs,axis=0)

        output_composition_string = "--".join([f"{tuple(a)}_{b}_{tuple(c)}" for a,b,c in output_composition])

        return {"sensor":sensor,
                "data":torch.Tensor(data),
                "rfs":torch.Tensor(rfs),
                "gsds":torch.Tensor(gsds),
                "output_composition":output_composition_string,
                "location":locations}
            
    def __len__(self):
        return len(self.locations)


if __name__ == "__main__":

    from omegaconf import OmegaConf
    import hydra
    cfg = OmegaConf.load('./configs/ev_MSViT_**.yaml')
    ds = hydra.utils.instantiate(cfg.dataset,train_val_test_key="train")
    item = ds.__getitem__(0)