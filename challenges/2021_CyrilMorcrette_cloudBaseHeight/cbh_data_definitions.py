import torch
import numpy as np
import zarr
import dask
import dask.array

# define torch dataloader
class CBH_Dataset(torch.utils.data.Dataset):
    def __init__(self, data_x, data_y, cloud_base_label):
        
        # print('begin init')
        
        self.temp_humidity_pressure = data_x
        self.cloudbase_target = data_y
        self.cbh_label = cloud_base_label
        
        self.height_layer_number = data_x.shape[1] # take the shape at index 1 as data_x of format sample, height, feature
        
        assert self.height_layer_number == 70
        
        # print('end init')
        
    def __len__(self):
        return len(self.temp_humidity_pressure)

    def __getitem__(self, idx):
        
        # since dask is being used, first compute the values on the index given to the get function, convert the array to tensor for pytorch
        
        # torch.from_numpy(x.compute())
        
        input_features = self.temp_humidity_pressure[idx]
        output_target = self.cloudbase_target[idx]
        # print(output_target.dtype)
        # output_target = output_target.type(torch.FloatTensor)
        cbh_lab = self.cbh_label[idx]
        
        # print('CALL ON GETITEM')
        
        height_vec = torch.from_numpy(np.arange(self.height_layer_number))
        
        item_in_dataset = {'x':input_features, 'cloud_volume_target':output_target, 'cloud_base_target':cbh_lab, 'height_vector':height_vec}
        return item_in_dataset
    
# load in the data from zarr, ensure correct chunk sizes
def load_data_from_zarr(path):
    
    store = zarr.DirectoryStore(path)
    zarr_group = zarr.group(store=store)
    print('Loaded zarr, file information:\n', zarr_group.info, '\n')
    
    x = dask.array.from_zarr(zarr_group['humidity_temp_pressure_x.zarr'])
    x.rechunk(zarr_group['humidity_temp_pressure_x.zarr'].chunks)
    
    y_lab = dask.array.from_zarr(zarr_group['onehot_cloud_base_height_y.zarr'])
    y_lab.rechunk(zarr_group['onehot_cloud_base_height_y.zarr'].chunks)
    
    y_cont = dask.array.from_zarr(zarr_group['cloud_volume_fraction_y.zarr'])
    y_cont.rechunk(zarr_group['cloud_volume_fraction_y.zarr'].chunks)
    
    return x, y_lab, y_cont

def define_data_get_loader(
                           inp, 
                           labels, 
                           vol_output,
                           batch_size, 
                           shuffle=False, 
                           num_workers = 0, 
                           collate_fn = None
                          ):
    dataset = CBH_Dataset(inp, labels, vol_output)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers = num_workers, collate_fn=collate_fn)
    return loader

# # define dask specific collate function for dataloader, collate is the step where the dataloader combines all the samples into a singular batch to be enumerated on, 
# # after getting all items 
def dataloader_collate_with_dask(batch):
    # print("call main collate")
    elem = batch[0]
    elem_type = type(elem)
    
    # assert torch.utils.data.get_worker_info() is None # if this assertion fails, there are issues in code and this case needs to be handled see pytorch source of default collate fn

    try:
        return elem_type({key: collate_helper_send_dict_elements_to_tensor([d[key] for d in batch]) for key in elem})
        
    except TypeError:
        # print('Should not have reached here')
        # raise TypeError()
        return {key: collate_helper_send_dict_elements_to_tensor([d[key] for d in batch]) for key in elem}
    
    raise TypeError(default_collate_err_msg_format.format(elem_type))

    
def collate_helper_send_dict_elements_to_tensor(batch):
    # assert torch.utils.data.get_worker_info() is None
    
    elem = batch[0]
    elem_type = type(elem)
    
    if elem_type is dask.array.core.Array:
        new_batch = np.stack(batch, 0) # emulate torch stack
        # print("Start compute", len(batch))
        new_batch = new_batch.compute()
        # print("End compute")
        return torch.from_numpy(new_batch)
        
    # elif isinstance(elem, torch.Tensor):
    #     out = None
    #     if torch.utils.data.get_worker_info() is not None:
    #         # If we're in a background process, concatenate directly into a
    #         # shared memory tensor to avoid an extra copy
    #         numel = sum(x.numel() for x in batch)
    #         storage = elem.storage()._new_shared(numel)
    #         out = elem.new(storage).resize_(len(batch), *list(elem.size()))
    #     return torch.stack(batch, 0, out=out)
    
    
    else:
        return torch.stack(batch, 0)
    
    raise TypeError(default_collate_err_msg_format.format(elem_type))

class CBH_Dataset_in_memory(torch.utils.data.Dataset):
    def __init__(self, data_x, data_y, cloud_base_label):
        
        data_x = data_x.compute()
        # data_y = data_y.compute()
        cloud_base_label = cloud_base_label.compute()
        
        self.temp_humidity_pressure = data_x
        self.cloudbase_target = data_y
        self.cbh_label = np.argmax(cloud_base_label, axis=1) ############################################TEMP############################################
        # print("init cbh label, size:",self.cbh_label.shape)
        
        self.height_layer_number = data_x.shape[1] # take the shape at index 1 as data_x of format sample, height, feature
        
        assert self.height_layer_number == 70
        
    def __len__(self):
        return len(self.temp_humidity_pressure)

    def __getitem__(self, idx):
        
        input_features = torch.from_numpy(self.temp_humidity_pressure[idx])
        output_target = [1] # torch.from_numpy(self.temp_humidity_pressure[idx])
        cbh_lab = torch.from_numpy(np.array([self.cbh_label[idx]]))
        
        height_vec = torch.from_numpy(np.arange(self.height_layer_number))
        
        item_in_dataset = {'x':input_features, 'cloud_volume_target':output_target, 'cloud_base_target':cbh_lab, 'height_vector':height_vec}
        return item_in_dataset



def define_data_get_loader_into_memory(
                           inp, 
                           labels, 
                           vol_output,
                           batch_size, 
                           shuffle=False, 
                           num_workers = 0, 
                           collate_fn = None
                          ):
    dataset = CBH_Dataset_in_memory(inp, labels, vol_output)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers = num_workers, collate_fn=None)
    return loader
