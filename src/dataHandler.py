import numpy as np
import torch
import utils

class MPII(torch.utils.data.Dataset):
  def __init__(self, path='../data/mpii_poses.npy'):
    super(MPII, self).__init__()
    self.poses = np.load(path)

  def __getitem__(self, key):
    if isinstance(key, slice):
      # get the start, stop, and step from the slice
      return [self[ii] for ii in range(*key.indices(len(self)))]
    elif isinstance(key, int):
      # handle negative indices
      if key < 0:
        key += len(self)
      if key < 0 or key >= len(self):
        raise IndexError("The index (%d) is out of range." % key)
      # get the data from direct index
      return self.get_item_from_index(key)
    else:
      raise TypeError("Invalid argument type.")

  def __len__(self):
    return self.poses.shape[0]

  def get_item_from_index(self, index):
    mpii_poses = self.poses[index:index+1]
    mpii_poses = utils.normalize_2d(mpii_poses)
    mpii_poses = mpii_poses.astype(np.float32)
    mpii_poses = torch.from_numpy(mpii_poses).squeeze(0)

    return mpii_poses
