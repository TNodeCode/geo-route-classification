import torch
import torch.nn as nn


class GeoRouteEmbedding(nn.Module):
    def __init__(self):
        super(GeoRouteEmbedding, self).__init__()
        # Non-sequential data
        self.embedding_src_as = nn.Embedding(5, 2)
        self.embedding_dest_as = nn.Embedding(52230, 16)
        self.embedding_src_cc = nn.Embedding(4, 2)
        self.embedding_dest_cc = nn.Embedding(231, 8)
        # Sequential data
        self.embedding_ip_source = nn.Embedding(5, 3)
        self.embedding_geo_cc = nn.Embedding(252, 8)
        self.embedding_asn = nn.Embedding(397771, 19)

    def get_src_as_embedding(self, x):
        """
        Get embedding for the srcAS sequences
        x       -- Raw srcAS features
        return  -- Embeddings for srcAS columns
        """
        # Run the tensor through an embedding layer
        return self.embedding_src_as(x)

    def get_dest_as_embedding(self, x):
        """
        Get embedding for the destAS sequences
        x       -- Raw destAS features
        return  -- Embeddings for destAS columns
        """
        # Run the tensor through an embedding layer
        return self.embedding_dest_as(x)

    def get_src_cc_embedding(self, x):
        """
        Get embedding for the srcCC sequences
        x       -- Raw srcCC features
        return  -- Embeddings for srcCC columns
        """
        # Run the tensor through an embedding layer
        return self.embedding_src_cc(x)

    def get_dest_cc_embedding(self, x):
        """
        Get embedding for the destCC sequences
        x       -- Raw destCC features
        return  -- Embeddings for destCC columns
        """
        # Run the tensor through an embedding layer
        return self.embedding_dest_cc(x)

    def get_non_sequential_embedding(self, x_src_as, x_dest_as, x_src_cc, x_dest_cc):
        """
        Get concatenated embeddings of all non-sequential features
        x_src_as    -- srcAS feature
        x_dest_as   -- destAS feature
        x_src_cc    -- srcCC feature
        x_dest_cc   -- destCC feature
        """
        return torch.cat([
            self.get_src_as_embedding(x_src_as),
            self.get_dest_as_embedding(x_dest_as),
            self.get_src_cc_embedding(x_src_cc),
            self.get_dest_cc_embedding(x_dest_cc),
        ], axis=2)

    def get_asn_embedding(self, x: torch.tensor):
        """
        Get embedding for the ASN sequences
        df      -- Pandas dataframe
        return  -- Embeddings for ASN columns
        """
        # Run the tensor through an embedding layer
        return self.embedding_asn(x)

    def get_geo_cc_embedding(self, x: torch.tensor):
        """
        Get embedding for the ASN sequences
        x       -- Raw ASN features
        return  -- Embeddings for geoCC columns
        """
        # Run the tensor through an embedding layer
        return self.embedding_geo_cc(x)

    def get_ip_source_embedding(self, x: torch.tensor):
        """
        Get embedding for the ASN sequences
        df      -- Pandas dataframe
        return  -- Embeddings for IPsource columns
        """
        # Run the tensor through an embedding layer
        return self.embedding_ip_source(x)

    def get_sequence_embedding(self, x_lat, x_long, x_asn, x_geo_cc, x_ip_source):
        """
        Get embedding for all sequential features ('ASN', 'geoCC', 'IPsource').
        This is done by concatenating embeddings all all sequential features.
        """
        return torch.cat([
            x_lat,
            x_long,
            self.get_asn_embedding(x_asn),
            self.get_geo_cc_embedding(x_geo_cc),
            self.get_ip_source_embedding(x_ip_source)
        ], axis=2)