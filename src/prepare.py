import os, sys, time, argparse
import pandas as pd
import polars as pl
from loguru import logger
from data_preparation import (
  load_dataset, categorize_columns, split_dataset,
  Imputer, FeatureScaler,
  landmarks_to_embedding,
  save_datasets, save_preprocessors
)


def prepare(combined=False, loso=False):
    t0 = time.time()
    logger.info(f"Starting data preparation process (combined={combined}, loso={loso})")

    df    = load_dataset("data/data_with_features.csv")
    logger.info(f"Loaded dataset with shape {df.shape[0]} rows")

    meta, landmark_cols, feature_cols = categorize_columns(df)
    logger.info(f"Number of landmark columns: {len(landmark_cols)}")
    logger.info(f"Number of engineered feature columns: {len(feature_cols)}")

    splits = split_dataset(df, combined=combined, loso=loso)
    for view, data in splits.items():
        logger.info(f"Processing {view} view dataset")
        if loso:
            full = data
            # …call embedding & saver for LOSO…
            continue

        train, test = data

        train_p = train.to_pandas()
        test_p  = test.to_pandas()

        logger.info(f"Train samples: {len(train_p)}, Test samples: {len(test_p)}")

        imp = Imputer()
        train_i = imp.fit_transform(train_p, feature_cols)
        test_i  = imp.transform(test_p, feature_cols)

        scaler = FeatureScaler()
        Xtr = scaler.fit_transform(train_i[feature_cols].to_numpy())
        Xte = scaler.transform (test_i [feature_cols].to_numpy())

        # create embeddings
        emb_tr = pd.DataFrame(train_i[landmark_cols].apply(landmarks_to_embedding, axis=1).tolist())
        emb_te = pd.DataFrame(test_i [landmark_cols].apply(landmarks_to_embedding, axis=1).tolist())

        # concat & save
        out_tr = pl.concat([pl.from_pandas(emb_tr), pl.from_pandas(Xtr)], how="horizontal")
        out_te = pl.concat([pl.from_pandas(emb_te), pl.from_pandas(Xte)], how="horizontal")

        save_datasets(out_tr, out_te, f"data/processed/{view}", view)
        save_preprocessors(scaler, imp, feature_cols, landmark_cols, f"data/processed/{view}")

    logger.success(f"Done in {time.time()-t0:.2f}s")


if __name__=="__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--combined", action="store_true")
    p.add_argument("--loso",     action="store_true")
    args = p.parse_args()
    prepare(combined=args.combined, loso=args.loso)
