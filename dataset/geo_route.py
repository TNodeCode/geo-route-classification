import pandas as pd
import torch
from torch.utils.data import Dataset


class GeoRouteDataset(Dataset):
    """
    This class loads data from a pandas dataframe
    """
    def __init__(self, filename: str):
        """
        Initialize the dataset class
        :param filename: The filename of the CSV file
        """
        df = pd.read_pickle(filename, compression='gzip').astype(float)
        # non-sequential data
        self.src_as = df[["srcAS_cat"]].to_numpy()
        self.dest_as = df[["destAS_cat"]].to_numpy()
        self.src_cc = df[["srcCC_cat"]].to_numpy()
        self.dest_cc = df[["destCC_cat"]].to_numpy()
        # sequrntial data
        self.lat = df.filter(regex="lat_*").to_numpy()
        self.long = df.filter(regex="long_*").to_numpy()
        self.asn = df.filter(regex="ASN_*").to_numpy()
        self.ip_source = df.filter(regex="IPsource_*").to_numpy()
        self.geo_cc = df.filter(regex="geoCC_*").to_numpy()
        # labels
        self.labels = df[["combined"]].to_numpy()

    def __len__(self):
        """
        This function returns the total number of items in the dataset.
        We are using a pandas data frame in this dataset which has an attribut named shape.
        The first dimension of shape is equal to the number of items in the dataset.
        :return: The number of rows in the CSV file
        """
        return self.src_as.shape[0]

    def __getitem__(self, idx):
        """
        This function returns a single tuple from the dataset.
        :param idx: The index of the tuple that should be returned.
        :return: Tuple of 
        """
        return self.src_as[idx], self.dest_as[idx], self.src_cc[idx], self.dest_cc[idx], self.lat[idx], self.long[idx], self.asn[idx], self.ip_source[idx], self.geo_cc[idx], self.labels[idx]

def prepare_tensors(src_as, dest_as, src_cc, dest_cc, lat, long, asn, ip_source, geo_cc, labels, device="cpu"):
    """
    This function takes the input tensors from the data loader and transforms them to the correct torch dtype (float32 | long)
    and also moves them to the correct device (cpu | cuda)
    """
    src_as = src_as.to(torch.long).to(device)
    dest_as = dest_as.to(torch.long).to(device)
    src_cc = src_cc.to(torch.long).to(device)
    dest_cc = dest_cc.to(torch.long).to(device)
    lat = lat.to(torch.float32).unsqueeze(dim=2).to(device)
    long = long.to(torch.float32).unsqueeze(dim=2).to(device)
    asn = asn.to(torch.long).to(device)
    ip_source = ip_source.to(torch.long).to(device)
    geo_cc = geo_cc.to(torch.long).to(device)
    labels = labels.to(torch.long).to(device)
    return src_as, dest_as, src_cc, dest_cc, lat, long, asn, ip_source, geo_cc, labels