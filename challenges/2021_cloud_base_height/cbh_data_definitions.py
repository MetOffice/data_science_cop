import torch
import pytorch_lightning as pl
import numpy as np
import zarr
import dask
import dask.array
from concurrent.futures import ThreadPoolExecutor
from dask.diagnostics import Profiler, ResourceProfiler, CacheProfiler, visualize
# cache=None
# def register_cache():
#     from dask.cache import Cache
#     global cache
#     cache = Cache(1e9)  # Leverage one gigabytes of memory (around 2 chunks)
#     cache.register()

# settings attempting to reduce unwanted mp
# from numcodecs import blosc
# blosc.use_threads = False
# dask.config.set(scheduler='synchronous')

THREAD_COUNT_FOR_DASK = 4
# define torch dataloader
class CBH_Dataset(torch.utils.data.Dataset):
    def __init__(self, data_x, cloud_base_label, thread_count_for_dask):
        
        # print('begin init')
        
        self.temp_humidity_pressure = data_x
        self.cbh_label = cloud_base_label
        global THREAD_COUNT_FOR_DASK
        THREAD_COUNT_FOR_DASK = thread_count_for_dask
        
        self.height_layer_number = data_x.shape[1] # take the shape at index 1 as data_x of format sample, height, feature
        
        assert self.height_layer_number == 70
        
        # print('end init')
        
    def __len__(self):
        # number of samples, is length of the input as input is shaped [sample,height,feat]
        return len(self.temp_humidity_pressure)

    def __getitem__(self, idx):
        
        # since dask is being used, first compute the values on the index given to the get function, convert the array to tensor for pytorch
        
        # torch.from_numpy(x.compute())
        
        input_features = self.temp_humidity_pressure[idx]
        cbh_lab = self.cbh_label[idx]
        
        # print('CALL ON GETITEM')
        return input_features, cbh_lab

# # define dask specific collate function for dataloader, collate is the step where the dataloader combines all the samples into a singular batch to be enumerated on, 
# # after getting all items 
def dataloader_collate_with_dask(batch):
    # print("call main collate")
    
    collated = tuple(dask.array.stack(groups) for groups in zip(*batch))
    input_batch, output_batch = collated
    global THREAD_COUNT_FOR_DASK
    with dask.config.set(pool=ThreadPoolExecutor(THREAD_COUNT_FOR_DASK)):#, Profiler() as prof, ResourceProfiler(dt=0.25) as rprof, CacheProfiler() as cprof:
        input_numpy = input_batch.compute()
        output_numpy = output_batch.compute()
    #print(visualize([prof, rprof, cprof], filename='profile.html', save=True))
    input_tensor = torch.from_numpy(input_numpy)
    output_tensor = torch.from_numpy(output_numpy)
    return input_tensor, output_tensor


class CBH_Dataset_in_memory(torch.utils.data.Dataset):
    def __init__(self, data_x, data_y, cloud_base_label, thread_count_for_dask, verbose=True):
        if verbose: print("Computing x...")
        data_x = data_x.compute()
        if verbose: print("Computing y...")
        data_y = data_y.compute()
        if verbose: print("Computing lab...")
        cloud_base_label = cloud_base_label.compute()
        global THREAD_COUNT_FOR_DASK
        THREAD_COUNT_FOR_DASK = thread_count_for_dask
        
        self.temp_humidity_pressure = data_x
        self.cloudbase_target = data_y
        self.cbh_label = cloud_base_label ############################################TEMP############################################
        # print("init cbh label, size:",self.cbh_label.shape)
        
        self.height_layer_number = data_x.shape[1] # take the shape at index 1 as data_x of format sample, height, feature
        
        assert self.height_layer_number == 70
        
    def __len__(self):
        return len(self.temp_humidity_pressure)

    def __getitem__(self, idx):
        
        input_features = torch.from_numpy(self.temp_humidity_pressure[idx])
        output_target = torch.from_numpy(self.cloudbase_target[idx])
        cbh_lab = self.cbh_label[idx]
        
        height_vec = torch.from_numpy(np.arange(self.height_layer_number))
        
        item_in_dataset = {'x':input_features, 'cloud_volume_target':output_target, 'cloud_base_target':cbh_lab, 'height_vector':height_vec}
        return item_in_dataset

class CBH_DataModule(pl.LightningDataModule):
    
    def __init__(self, 
                 tr_data_x, tr_cloud_base_label, 
                 val_data_x, val_cloud_base_label, 
                 thread_count_for_dask,
                 batch_size,
                 num_workers = 0,
                 collate_fn = None,
                 shuffle = False,
                 randomize_chunkwise = False,
                 method='1chunk',
                ):
        super().__init__()
        
        self.method=method
        
        global THREAD_COUNT_FOR_DASK
        THREAD_COUNT_FOR_DASK = thread_count_for_dask
        self.dataloader_workers = num_workers
        
        self.batch_size = batch_size
        self.shuffle_train = shuffle
        self.collate_fn = collate_fn
        
        self.train_data_x = tr_data_x
        self.train_data_y = tr_cloud_base_label
        self.randomize_chunkwise = randomize_chunkwise
        
        self.val_data_x = val_data_x
        self.val_data_y = val_cloud_base_label
        
    def setup(self, stage):
        if(self.method == "1chunk"):
            self.train_dset = CBH_Dataset_Load_One_Chunk(
                self.train_data_x,
                self.train_data_y,
                self.randomize_chunkwise
            )
            self.val_dset = CBH_Dataset_Load_One_Chunk(
                self.val_data_x,
                self.val_data_y,
                self.randomize_chunkwise
            )
        else:
            raise Exception("Not Implemented Error")
            
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dset, batch_size=self.batch_size, shuffle=self.shuffle_train, num_workers = self.dataloader_workers, collate_fn=self.collate_fn)
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dset, batch_size=self.batch_size, shuffle=self.shuffle_train, num_workers = self.dataloader_workers, collate_fn=self.collate_fn)

class CBH_Dataset_Load_One_Chunk(torch.utils.data.Dataset):
    def __init__(self, data_x, cloud_base_label, randomize_chunkwise = False, threads=None):
        
        # print('begin init')
        
        self.temp_humidity_pressure = data_x
        self.randomize_chunkwise = randomize_chunkwise
        self.cbh_label = cloud_base_label
        if threads!=None:
            global THREAD_COUNT_FOR_DASK
            THREAD_COUNT_FOR_DASK = threads
        
        self.height_layer_number = data_x.shape[1] # take the shape at index 1 as data_x of format sample, height, feature
        
        assert self.height_layer_number == 70
        
        # self.loaded_chunk_x = None
        # self.loaded_chunk_y = None
        self.chunk_size = data_x.chunksize[0]
        
        self.maxidx = 0
        
        #init first chunks
        self.loaded_chunk_x = self.temp_humidity_pressure[0:self.chunk_size].compute()
        self.loaded_chunk_y = self.cbh_label[0:self.chunk_size].compute()
            
        # print('end init')
        
    def __len__(self):
        # number of samples, is length of the input as input is shaped [sample,height,feat]
        return len(self.temp_humidity_pressure)

    def _compute_and_store_chunk(self, idx):
        if self.maxidx == len(self):
            self.maxidx = 0
        next_max_id = self.maxidx + self.chunk_size
        # print("maxid", self.maxidx)
        # print("next_max_id", next_max_id)
        with dask.config.set(pool=ThreadPoolExecutor(THREAD_COUNT_FOR_DASK)):#, Profiler() as prof, ResourceProfiler(dt=0.25) as rprof, CacheProfiler() as cprof:
            self.loaded_chunk_x = self.temp_humidity_pressure[self.maxidx:next_max_id].compute()
            self.loaded_chunk_y = self.cbh_label[self.maxidx:next_max_id].compute()
            # print("Done compute()")
            # print("Type of obj:", type(self.loaded_chunk_x))
        
        if self.randomize_chunkwise:
            p = np.random.permutation(len(self.loaded_chunk_y))
            self.loaded_chunk_y = self.loaded_chunk_y[p]
            self.loaded_chunk_x = self.loaded_chunk_x[p]
        
        self.maxidx = next_max_id
    
    def __getitem__(self, idx):
        
        # since dask is being used, first compute the values on the index given to the get function, convert the array to tensor for pytorch
        
        # print('idx', idx)
        
        if idx >= self.maxidx:
            # print("call compute")
            self._compute_and_store_chunk(idx)

        input_features = self.loaded_chunk_x[int(idx % self.chunk_size),:,:]
        cbh_lab = self.loaded_chunk_y[int(idx % self.chunk_size)]
        
        # print('CALL ON GETITEM')
        return input_features, cbh_lab

# load in the data from zarr, ensure correct chunk sizes
def load_data_from_zarr(path, get_cont=False):
    
    store = zarr.DirectoryStore(path)
    zarr_group = zarr.group(store=store, synchronizer=zarr.sync.ThreadSynchronizer())
    print('Loaded zarr, file information:\n', zarr_group.info, '\n')
    
    x = dask.array.from_zarr(zarr_group['humidity_temp_pressure_x.zarr'])
    x.rechunk(zarr_group['humidity_temp_pressure_x.zarr'].chunks)
    
    y_lab = dask.array.from_zarr(zarr_group['cloud_base_label_y.zarr'])
    y_lab.rechunk(zarr_group['cloud_base_label_y.zarr'].chunks)
    
    y_cont=None
    if get_cont:
        y_cont = dask.array.from_zarr(zarr_group['cloud_volume_fraction_y.zarr'])
        y_cont.rechunk(zarr_group['cloud_volume_fraction_y.zarr'].chunks)
    
    return x, y_lab, y_cont