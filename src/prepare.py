import os
import time
import argparse

import pandas as pd

from loguru import logger

from data_preparation.imputer import Imputer
from data_preparation.scaler import FeatureScaler
from data_preparation.embedding import landmarks_to_embedding
from data_preparation.saver import save_datasets, save_preprocessors
from data_preparation.splitter import split_dataset
from data_preparation.data_filter import filter_dataset, filter_features


def prepare(
    combined: bool = False,
    loso: bool = False,
    data_mode: str = "all",
    feature_mode: str = "all_features",
    force_real_test: bool = False,
):
    t0 = time.time()
    logger.info(
        f"Starting data preparation (combined={combined}, loso={loso}, "
        f"data_mode={data_mode}, feature_mode={feature_mode}, force_real_test={force_real_test})"
    )

    # Load the enriched landmarks data (with features)
    data_in = "data/data_with_features.csv"
    if not os.path.exists(data_in):
        raise FileNotFoundError(
            f"{data_in} not found. Please run feature extraction first."
        )
    df = pd.read_csv(data_in)
    logger.info(
        f"Loaded raw dataset with shape {df.shape[0]} rows and {df.shape[1]} columns"
    )

    # Filter dataset based on ablation setting
    df = filter_dataset(df, mode=data_mode)
    logger.info(f"After filtering ({data_mode}): {df.shape[0]} rows")

    # Filter features based on ablation setting (pandas)
    meta_cols, landmark_cols, feature_cols = filter_features(
        df, feature_mode=feature_mode
    )
    logger.info(
        f"Feature filtering ({feature_mode}): "
        f"{len(landmark_cols)} landmark cols, {len(feature_cols)} feature cols"
    )

    # Split dataset
    splits = split_dataset(df, combined=combined)

    for view, data in splits.items():
        logger.info(f"Processing '{view}' view dataset")

        train_df, test_df = data

        # For ablation study: when testing, always use real data only if forced
        if data_mode != "real" and force_real_test:
            test_df = filter_dataset(test_df, mode="real")
            logger.info(f"Filtered test set to real-only data: {len(test_df)} samples")

        train_p = train_df.copy()
        test_p = test_df.copy()

        logger.info(f"Train samples: {len(train_p)}, Test samples: {len(test_p)}")

        # Process engineered features if present
        scaler = None
        imp = None
        Xtr_df = pd.DataFrame(index=train_p.index)
        Xte_df = pd.DataFrame(index=test_p.index)

        if feature_cols:
            imp = Imputer()
            train_i = imp.fit_transform(train_p, feature_cols)
            test_i = imp.transform(test_p, feature_cols)

            scaler = FeatureScaler()
            Xtr = scaler.fit_transform(train_i[feature_cols].to_numpy())
            Xte = scaler.transform(test_i[feature_cols].to_numpy())

            Xtr_df = pd.DataFrame(
                Xtr,
                index=train_i.index,
                columns=feature_cols
            )
            Xte_df = pd.DataFrame(
                Xte,
                index=test_i.index,
                columns=feature_cols
            )
        else:
            train_i = train_p
            test_i = test_p

        # Create embeddings
        emb_tr = pd.DataFrame(
            train_i[landmark_cols].apply(landmarks_to_embedding, axis=1).tolist(),
            index=train_i.index,
            columns=[col for col in landmark_cols if not col.endswith("_score")]
        )
        emb_te = pd.DataFrame(
            test_i[landmark_cols].apply(landmarks_to_embedding, axis=1).tolist(),
            index=test_i.index,
            columns=[col for col in landmark_cols if not col.endswith("_score")]
        )

        # Combine embeddings + engineered features
        out_tr = pd.concat([emb_tr, Xtr_df], axis=1) if not Xtr_df.empty else emb_tr
        out_te = pd.concat([emb_te, Xte_df], axis=1) if not Xte_df.empty else emb_te

        # Add labels column for training; prefer class_no if available
        if "class_no" in train_i.columns:
            out_tr["labels"] = train_i["class_no"].values
        else:
            raise KeyError("Neither 'class_no' nor 'class_name' found to build labels.")

        if "class_no" in test_i.columns:
            out_te["labels"] = test_i["class_no"].values
        else:
            raise KeyError(
                "Neither 'class_no' nor 'class_name' found to build labels for test set."
            )

        # Save with data mode and feature mode in path
        save_path = os.path.join("data/processed", f"{data_mode}_{feature_mode}")
        save_datasets(out_tr, out_te, save_path, view)
        # for loso
        final_df_indexes = train_p.index.to_list() + test_p.index.to_list()
        pd.concat([
            pd.concat([out_tr, out_te]),
            df.loc[final_df_indexes, meta_cols],
        ], axis=1).to_csv(os.path.join(save_path, view, "data.csv"), index=False)
        logger.info(f"Data saved to {save_path}")

        # Save preprocessors if they exist
        if scaler is not None or imp is not None:
            save_preprocessors(scaler, imp, feature_cols, landmark_cols, save_path)

    logger.success(f"Data preparation completed in {time.time() - t0:.2f}s")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--combined", action="store_true")
    p.add_argument("--loso", action="store_true")
    p.add_argument(
        "--data-mode",
        type=str,
        choices=["all", "real", "synthetic"],
        default="all",
        help="Data mode for ablation study: all (real+synthetic), real-only, or synthetic-only",
    )
    p.add_argument(
        "--feature-mode",
        type=str,
        choices=["all_features", "keypoints_only"],
        default="all_features",
        help="Feature mode for ablation study: all_features (keypoints+engineered) or keypoints_only",
    )
    p.add_argument(
        "--force-real-test",
        action="store_true",
        help="Force the test set to be real-only data",
    )
    args = p.parse_args()
    prepare(
        combined=args.combined,
        loso=args.loso,
        data_mode=args.data_mode,
        feature_mode=args.feature_mode,
        force_real_test=args.force_real_test,
    )
