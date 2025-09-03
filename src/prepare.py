import os
import time
import argparse
import pandas as pd
import polars as pl
from loguru import logger
from data_preparation import (
  load_dataset, split_dataset,
  Imputer, FeatureScaler,
  landmarks_to_embedding,
  save_datasets, save_preprocessors
)
from data_preparation.data_filter import filter_dataset, filter_features


def prepare(combined=False, loso=False, data_mode='all', feature_mode='all_features'):
    t0 = time.time()
    logger.info(f"Starting data preparation process (combined={combined}, loso={loso}, data_mode={data_mode}, feature_mode={feature_mode})")

    df = load_dataset("data/data_with_features.csv")
    logger.info(f"Loaded raw dataset with shape {df.shape[0]} rows")

    # Filter dataset based on ablation setting
    df = filter_dataset(df, mode=data_mode)
    logger.info(f"After filtering ({data_mode}): {df.shape[0]} rows")

    # Filter features based on ablation setting
    meta_cols, landmark_cols, feature_cols = filter_features(df, feature_mode=feature_mode)
    logger.info(f"Feature filtering ({feature_mode}): {len(landmark_cols)} landmark cols, {len(feature_cols)} feature cols")

    splits = split_dataset(df, combined=combined, loso=loso)
    for view, data in splits.items():
        logger.info(f"Processing {view} view dataset")
        if loso:
            full = data
            # Process LOSO data
            full_p = full.to_pandas()

            # Apply imputation if we have feature columns
            if feature_cols:
                imp = Imputer()
                full_i = imp.fit_transform(full_p, feature_cols)

                # Scale features
                scaler = FeatureScaler()
                Xf = scaler.fit_transform(full_i[feature_cols].to_numpy())
                Xf_df = pd.DataFrame(Xf, columns=[f"feat_{i}" for i in range(Xf.shape[1])])
            else:
                # No feature columns, just create empty DataFrames
                Xf_df = pd.DataFrame()
                imp = None
                scaler = None

            # Create embeddings from landmarks
            emb_f = pd.DataFrame(full_p[landmark_cols].apply(landmarks_to_embedding, axis=1).tolist())

            # Combine data
            if len(Xf_df) > 0:
                out_f = pl.concat([pl.from_pandas(emb_f), pl.from_pandas(Xf_df)], how="horizontal")
            else:
                out_f = pl.from_pandas(emb_f)

            # Add metadata
            out_f = pl.concat([
                full.select(pl.col(c) for c in meta_cols if c in full.columns),
                out_f
            ], how="horizontal")
#
            # Save the data
            save_path = f"data/processed/{data_mode}_{feature_mode}/{view}"
            os.makedirs(save_path, exist_ok=True)
            out_f.write_csv(f"{save_path}/data.csv")

            # Save preprocessors if they exist
            if scaler and imp:
                save_preprocessors(scaler, imp, feature_cols, landmark_cols, save_path)

            continue

        train, test = data

        # For ablation study: when testing, always use real data only
        if data_mode != 'real':
            test = filter_dataset(test, mode='real')
            logger.info(f"Filtered test set to real-only data: {len(test)} samples")

        train_p = train.to_pandas()
        test_p  = test.to_pandas()

        logger.info(f"Train samples: {len(train_p)}, Test samples: {len(test_p)}")

        # Process features if we have them
        if feature_cols:
            imp = Imputer()
            train_i = imp.fit_transform(train_p, feature_cols)
            test_i  = imp.transform(test_p, feature_cols)

            scaler = FeatureScaler()
            Xtr = scaler.fit_transform(train_i[feature_cols].to_numpy())
            Xte = scaler.transform(test_i[feature_cols].to_numpy())

            Xtr_df = pd.DataFrame(Xtr, columns=[f"feat_{i}" for i in range(Xtr.shape[1])])
            Xte_df = pd.DataFrame(Xte, columns=[f"feat_{i}" for i in range(Xte.shape[1])])
        else:
            # No feature columns, use unmodified data and empty DataFrames
            train_i = train_p
            test_i = test_p
            Xtr_df = pd.DataFrame(index=train_i.index)
            Xte_df = pd.DataFrame(index=test_i.index)
            imp = None
            scaler = None

        # create embeddings
        emb_tr = pd.DataFrame(train_i[landmark_cols].apply(landmarks_to_embedding, axis=1).tolist())
        emb_te = pd.DataFrame(test_i[landmark_cols].apply(landmarks_to_embedding, axis=1).tolist())

        # concat & save
        if len(Xtr_df.columns) > 0:
            out_tr = pl.concat([pl.from_pandas(emb_tr), pl.from_pandas(Xtr_df)], how="horizontal")
            out_te = pl.concat([pl.from_pandas(emb_te), pl.from_pandas(Xte_df)], how="horizontal")
        else:
            out_tr = pl.from_pandas(emb_tr)
            out_te = pl.from_pandas(emb_te)

        # Save with data mode and feature mode in path
        save_path = f"data/processed/{data_mode}_{feature_mode}/{view}"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        save_datasets(out_tr, out_te, save_path, view)

        # Save preprocessors if they exist
        if scaler and imp:
            save_preprocessors(scaler, imp, feature_cols, landmark_cols, save_path)

    logger.success(f"Done in {time.time()-t0:.2f}s")


if __name__=="__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--combined", action="store_true")
    p.add_argument("--loso",     action="store_true")
    p.add_argument("--data-mode", type=str, choices=['all', 'real', 'synthetic'], default='all',
                   help="Data mode for ablation study: all (real+synthetic), real-only, or synthetic-only")
    p.add_argument("--feature-mode", type=str, choices=['all_features', 'keypoints_only'], default='all_features',
                   help="Feature mode for ablation study: all_features (keypoints+engineered) or keypoints_only")
    args = p.parse_args()
    prepare(combined=args.combined, loso=args.loso, data_mode=args.data_mode, feature_mode=args.feature_mode)
