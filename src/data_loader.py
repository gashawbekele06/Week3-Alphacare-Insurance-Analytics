import pandas as pd
import logging
from src.config import DATA_PATH

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InsuranceDataLoader:
    def load(self):
        if not DATA_PATH.exists():
            raise FileNotFoundError(f"Data not found: {DATA_PATH}")

        for sep in ["|", "\t", ",", ";"]:
            try:
                df = pd.read_csv(DATA_PATH, sep=sep, low_memory=False, nrows=1000)
                if df.shape[1] > 20:
                    logger.info(f"Delimiter found: '{sep}'")
                    df = pd.read_csv(DATA_PATH, sep=sep, low_memory=False)
                    logger.info(f"Loaded {df.shape[0]:,} rows")
                    
                    # Safe type conversion
                    if 'TransactionMonth' in df.columns:
                        df['TransactionMonth'] = pd.to_datetime(df['TransactionMonth'], errors='coerce')
                    for col in ['TotalPremium', 'TotalClaims']:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    return df
            except:
                continue
        raise ValueError("Could not read file")