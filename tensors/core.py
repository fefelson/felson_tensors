from copy import deepcopy
import numpy as np
import os 
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Sampler, Subset

from ..database.models.database import get_db_session
from ..utils.logging_manager import get_logger


######################################################################
######################################################################

BASE_PATH = os.environ["HOME"] + "/FEFelson/leagues"

######################################################################
######################################################################

logger = get_logger()

class BaseModel(nn.Module):

    _entityType = None 
    _modelName = None
    _leagueId = None  

    def __init__(self, *, entityId: str, defaultId: str=None):
        super().__init__()
        print(f"\ntensors.core:31 BaseModel- entityId:{entityId}, defaultId:{defaultId}, modelName: {self._modelName}")
        self.entityId = entityId
        self.defaultId = defaultId
        self.metrics = {}


    def _load(self):

        entityPath = os.path.join(BASE_PATH, self._leagueId, "simulations", self._entityType, self.entityId, f"{self._modelName}.pt")
        if not os.path.exists(entityPath) and self.defaultId is not None:
            logger.debug("PATH NOT FOUND -- USING DEFAULT")
            entityPath =  os.path.join(BASE_PATH, self._leagueId, "simulations", self._entityType, self.defaultId, f"{self._modelName}.pt")
        try:
            self.load_state_dict(torch.load(entityPath))
        except FileNotFoundError:
            logger.debug("DEFAULT NOT FOUND -- CREATING DEFAULT")


    def _save(self, metrics):
        
        if metrics["loss"] < self.metrics.get("loss", float("inf")):
            self.metrics = metrics

            modelPath = os.path.join(BASE_PATH, self._leagueId, "simulations", self._entityType, self.entityId, f"{self._modelName}.pt")
            os.makedirs(os.path.dirname(modelPath), exist_ok=True)        
            torch.save(self.state_dict(), modelPath)
            # logger.debug("saved model")


    def _exists(self):
        modelPath = os.path.join(BASE_PATH, self._leagueId, "simulations", self._entityType, self.entityId, f"{self._modelName}.pt")
        return os.path.exists(modelPath)



######################################################################
######################################################################


class ContinuousEmbed(nn.Module):
    def __init__(self, in_dim=1, out_dim=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )

    def forward(self, x):
        return self.fc(x)


######################################################################
######################################################################


class CustomDataset(Dataset):

    _main_table = None
    _main_id = None
    _featured_id = None
    _joins = []
    _select_features = None
    _categorical_features = None
    _numeric_features = None
    _conditionStmt = None
    _features = {"batter": [], "pitcher": [], "pitch":[], "hit":[], "count":[]}
    _label = None

    def __init__(self, *, entityId: str=None, condition: str=None):
        """
        Initialize the dataset with entity ID and batch size for database queries.
        """
        self.entityId = entityId
        self.condition = condition
        self.cache = None
        
        self.means = self._set_computation("AVG")
        self.stds = self._set_computation("STDDEV")
        
        self.valid_indicies = self._set_valid_indices()


    def __len__(self):
        """Return the total number of records in the dataset."""
        return len(self.valid_indicies)



    def __getitem__(self, idx: int):
        """
        Fetch a single data item by index, using cached batches.

        Returns:
            tuple: (features, label), where features is a nested dict of tensors and label is a tensor.
        """

        row = self.cache[self.cache[self._main_id] == idx].iloc[0] 
        features = {}
        for ft_key, ft_values in self._features.items():
            feature_types = {}
            if not ft_values:
                continue
            for ftr in self._select_stmt(ft_values):
                # print("_get_item_:136",ftr, row[ftr["ftr"]])
                dtype = torch.float32 if ftr["ftr"] in [ftr["ftr"] for ftr in self._select_stmt(self._numeric_features)] else torch.long
                feature_types[ftr["ftr"]] = torch.tensor(row[ftr["ftr"]], dtype=dtype)
            features[ft_key] = deepcopy(feature_types)
        # print(row)
        
        dtype = torch.float32 if self._label in [ftr["ftr"] for ftr in self._select_stmt(self._numeric_features)] else torch.long
        labels = torch.tensor(row[self._label], dtype=dtype)
        return features, labels


    def _fetch_db_batch(self, rowids: list):
        selectClause = ", ".join(ftr for ftr in self._select_features)
        fromStmt = self._from_stmt()
        joinStmt = self._join_stmt()
        filterClause = f"AND {self._filter_clause()}" if self._filter_clause() else "" 
        whrStmt = f"WHERE {self._main_id} IN {tuple(rowids)} {filterClause}" if len(rowids) > 1 else f"WHERE {self._main_id} = {rowids[0]} {filterClause}"
        
        query = f"""SELECT {self._main_id},
                            {selectClause}
                    {fromStmt}
                    {joinStmt}
                    {whrStmt}
                    ORDER BY {self._main_id}
                    """
        # print(query)
        # raise
        with get_db_session() as session:
            df = pd.read_sql(query, session.bind)
        self.cache = self._preprocess_batch(df)


    def _filter_clause(self):
        eCommand = f"{self._featured_id} = '{self.entityId}'" if self.entityId else None
        cCommand = self._conditionStmt if self._conditionStmt else None
        stmt = " AND ".join(cmd for cmd in (eCommand, cCommand, self.condition) if cmd) if eCommand or cCommand or self.condition else ""
        return stmt

    
    def _from_stmt(self):
        frmTbl = self._main_table
        abrv = self._table_abrv(frmTbl)
        return f"FROM {frmTbl} AS {abrv}" 


    def _join_stmt(self):
        joinStmt = ""
        for table in self._joins:
            name = table["name"].split("AS")[0]
            abrv = self._table_abrv(table["name"])
            joinType = "INNER" if table["inner"] else "LEFT"
            joinDetails = " AND ".join(f"{abrv}.{jt['keys'][0]} = {self._table_abrv(jt['name'])}.{jt['keys'][-1]}" for jt in table["join_tables"] )
            tableStmt = f"\t{joinType} JOIN {name} AS {abrv} ON {joinDetails} \n"
            joinStmt += tableStmt 
        # print(joinStmt)
        # raise                 
        return joinStmt 


    def _preprocess_batch(self, df):
        """Transform batch with Pandas."""
        # print(df.head(10))
        for ftr in self._select_stmt(self._numeric_features):
            col = ftr["ftr"]
            # Ensure the column is of float dtype
            df[col] = df[col].astype('float64')
            # Set missing values to the mean
            df.loc[df[col].isna(), col] = self.means[col]
            # Normalize
            df[col] = (df[col] - self.means[col]) / self.stds[col]

        for ftr in self._select_stmt(self._categorical_features):
            col = ftr["ftr"]
            col = col.split(".")[-1]
            # Convert to numeric, coercing errors to NaN, then to nullable integer
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
            # Set NaN values to mode
            mode = df[col].dropna().mode()
            df.loc[df[col].isna(), col] = mode.iloc[0]

        # print(df.head(10))
        # raise
        return df


    def _select_stmt(self, features):
        stmts = []
        for ftr in features:
            if "AS" not in ftr:
                stmts.append({"stmt":ftr, "ftr":ftr})
            else:
                stmt,field = [f.strip() for f in ftr.split("AS")]
                stmts.append({"stmt":stmt, "ftr":field})
                
        return stmts
    

    def _set_computation(self, func: str):
        """
        Compute statistics (e.g., AVG, STDDEV) for numeric features.
        Returns:
            dict: Mapping of feature names to computed values.
        """
        selectStmt = ", ".join(f"{func}({ftr['stmt']}) AS {ftr['ftr']}" for ftr in self._select_stmt(self._numeric_features))
        fromStmt = self._from_stmt()
        joinStmt = self._join_stmt()
        
        query = f"""
            SELECT {selectStmt}
            {fromStmt}
            {joinStmt}
        """
        # print(query)
        # raise
        with get_db_session() as session:
            data = pd.read_sql(query, session.bind)
        return {ftr['ftr']: data[ftr['ftr']][0] for ftr in self._select_stmt(self._numeric_features)}


    def _set_valid_indices(self):
        fromStmt = self._from_stmt()
        joinStmt = self._join_stmt()
        whereStmt = f"WHERE {self._filter_clause()}" if self._filter_clause() else ""

        query = f"""
                SELECT {self._main_id}
                {fromStmt}
                {joinStmt}
                {whereStmt}
                ORDER BY {self._main_id}
                """
        # print(query, end="\n\n")
        # raise
        with get_db_session() as session:
            return pd.read_sql(query, session.bind)[self._main_id].tolist()


    def _table_abrv(self, tableName):
        if "AS" not in tableName:
            abrv = "".join([c[0] for c in tableName.split("_")])
        else:
            abrv = tableName.split("AS")[-1].strip()
        return abrv



######################################################################
######################################################################



class RegisteredEntity:

    _class_map = None 
    
    def get_registered_entity(obj, entity_id):

        models = {}
        for name, cls_type in obj.class_map.items():
            models[name] = deepcopy(cls_type(entity_id))
        return models



######################################################################
######################################################################


def get_data_loader(dataset, *, batch_size = 64, db_batch_size= 6400, shuffle=False):
    """Create a DataLoader for the given dataset split."""

    class CacheAwareBatchSampler(Sampler):
        def __init__(self, subset: Subset, batch_size: int, cache_size: int, shuffle: bool):
            self.subset = subset
            self.batch_size = batch_size
            self.cache_size = cache_size
            self.shuffle = shuffle

        def __iter__(self):
            # if self.shuffle:
            #     # Shuffle indices within each db_batch_size chunk
            #     for i in range(0, len(self.dataset), self.db_batch_size):
            #         chunk = self.indices[i:i + self.db_batch_size]
            #         np.random.shuffle(chunk)
            #         self.indices[i:i + self.db_batch_size] = chunk

            for i in range(0, len(self.subset), self.cache_size):
                cache_size = self.cache_size if i+self.cache_size < len(self.subset) else len(self.subset) - i
                chunk = self.subset.indices[i:i +cache_size]
                self.subset.dataset._fetch_db_batch(chunk)
                
                for j in range(i, i+cache_size, self.batch_size):
                    batch_size = self.batch_size if (j-i)+self.batch_size <= cache_size else cache_size - (j-i)
                    yield list(range(j,j+batch_size))
                    

        def __len__(self):
            return len(self.subset) // self.batch_size


    def collate_fn(batch):
        features_batch, labels = zip(*batch)

        features = {}
        for key, value in features_batch[0].items():
            for ftr in value.keys():
                features[ftr] = torch.stack([fb[key][ftr] for fb in features_batch])
        labels = torch.stack([l for l in labels])
        # from pprint import pprint 
        # pprint(features)
        # print()
        # pprint(labels)
        # raise
        return features, labels

    sampler = CacheAwareBatchSampler(dataset, batch_size, db_batch_size, shuffle)
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collate_fn
    )
    return loader



def split_dataset(dataset: CustomDataset, *, train_ratio: float=0.7, val_ratio: float=0.2, test_ratio: float=0.1, seed: int=42):
    
    rng = np.random.RandomState(seed)
    dataset_size = len(dataset)

    indices = np.array(dataset.valid_indicies)

    n_train = int(len(indices) * train_ratio)
    n_val = int(len(indices) * val_ratio)
    n_test = len(indices) - n_train - n_val

    rng.shuffle(indices)

    train_indices = indices[:n_train].tolist()
    val_indices = indices[n_train:n_train + n_val].tolist()
    test_indices = indices[n_train + n_val:].tolist()

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    return train_dataset, val_dataset, test_dataset
