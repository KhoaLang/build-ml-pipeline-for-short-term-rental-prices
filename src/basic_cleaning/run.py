#!/usr/bin/env python
"""
Performs basic cleaning on the data and save the results in Weights & Biases
"""
import argparse
import logging
import wandb
import pandas as pd


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    logger.info("Downloading artifacts...")
    artifact = run.use_artifact(args.input_artifact)
    artifact_path = artifact.file()

    logger.info('Drop duplicates...')
    df = pd.read_csv(artifact_path)
    df.drop_duplicates().reset_index(drop=True)

    # Drop outliers
    logger.info("Dropping outliers")
    logger.info(f"Number of rows before dropping outliers: {df.shape[0]}")
    idx = df['price'].between(args.min_price, args.max_price)
    df = df[idx].copy()
    idx = df['longitude'].between(-74.25, -73.50) & df['latitude'].between(40.5, 41.2)
    df = df[idx].copy()
    logger.info(f"Number of rows after dropping outliers: {df.shape[0]}")

    # Convert last_review to datetime
    logger.info("Converting last_review to datetime")
    df['last_review'] = pd.to_datetime(df['last_review'])

    # Save the dataframe to a csv file and log it as an artifact
    logger.info("Saving cleaned dataframe to csv")
    df.to_csv(args.output_artifact, index=False)

    artifact = wandb.Artifact(
        name=args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    artifact.add_file(args.output_artifact)

    logger.info("Logging artifact")
    run.log_artifact(artifact)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="This steps cleans the data")


    parser.add_argument(
        "--input_artifact",
        type=str,
        help='Get the artifact from Weight&Bias',
        required=True
    )
    
    parser.add_argument(
        "--output_artifact", 
        type=str,
        help='Get the artifact from Weight&Bias',
        required=True
    )
    
    parser.add_argument(
        "--output_type", 
        type=str,
        help='Output type',
        required=True
    )
    
    parser.add_argument(
        "--output_description", 
        type=str,
        help='Output description',
        required=True
    )

    parser.add_argument(
        "--min_price", 
        type=float,
        help='Min price to filter out outlier',
        required=True
    )

    parser.add_argument(
        "--max_price", 
        type=float,
        help='Max price to filter out outlier',
        required=True
    )


    args = parser.parse_args()

    go(args)
