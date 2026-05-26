import os
import pandas as pd
from fastapi import APIRouter, HTTPException
from fastapi_cache.decorator import cache
from pydantic import BaseModel

from backend.config import DATA_DIR, setup_logger
from models.absa_model import AspectSentimentAnalyzer
from models.summary_model import ReviewSummarizer
from models.aspect_extractor import KMeansAspectExtractor

router = APIRouter()
logger = setup_logger("APIRoutes")

absa_model = AspectSentimentAnalyzer()
summary_model = ReviewSummarizer()
aspect_extractor = KMeansAspectExtractor(n_clusters=4)

ds1_path = os.path.join(DATA_DIR, "dataset1.parquet")
ds2_path = os.path.join(DATA_DIR, "dataset2.parquet")
df1 = pd.read_parquet(ds1_path) if os.path.exists(ds1_path) else pd.DataFrame()
df2 = pd.read_parquet(ds2_path) if os.path.exists(ds2_path) else pd.DataFrame()

logger.info(f"Loaded DataFrame 1 shape: {df1.shape}")
logger.info(f"Loaded DataFrame 2 shape: {df2.shape}")


class AspectDetail(BaseModel):
    pos: float
    neg: float
    neu: float
    mentions: int


class AspectResponse(BaseModel):
    movie_id: str
    aspects: dict[str, AspectDetail]


class SummaryResponse(BaseModel):
    movie_id: str
    summary: str
    average_rating: float | None


@router.get("/api/v1/movie/{movie_id}/aspects", response_model=AspectResponse)
@cache(expire=86400)
async def get_movie_aspects(movie_id: str):
    logger.info(f"Received aspects request for movie_id: {movie_id}")

    if df1.empty:
        logger.error("Dataset1 is empty or not loaded")
        raise HTTPException(status_code=500, detail="Dataset not loaded")

    movie_data = df1[df1["movie_id"] == movie_id]

    logger.info(f"Filtered DataFrame shape for movie_id {movie_id}: {movie_data.shape}")

    if movie_data.empty:
        logger.warning(f"No data found for movie_id: {movie_id}")
        raise HTTPException(status_code=404, detail="Movie not found")

    texts = movie_data["review_text"].tolist()

    logger.info(f"Extracted {len(texts)} review texts to list")

    generated_aspects = aspect_extractor.fit_predict(texts)

    logger.info(f"Dynamically generated aspects for movie_id {movie_id}: {generated_aspects}")

    aspect_scores = absa_model.predict(texts, generated_aspects)

    logger.info(f"Aspect scores calculation completed for movie_id {movie_id}")

    return AspectResponse(movie_id=movie_id, aspects=aspect_scores)


@router.get("/api/v1/movie/{movie_id}/summary", response_model=SummaryResponse)
@cache(expire=86400)
async def get_movie_summary(movie_id: str):
    logger.info(f"Received summary request for movie_id: {movie_id}")

    if df2.empty or df1.empty:
        logger.error("Datasets are empty or not loaded")
        raise HTTPException(status_code=500, detail="Dataset not loaded")

    movie_data_summ = df2[df2["movie_id"] == movie_id]
    movie_data_rating = df1[df1["movie_id"] == movie_id]

    logger.info(
        f"Filtered summary DataFrame shape: {movie_data_summ.shape}, "
        f"rating DataFrame shape: {movie_data_rating.shape}"
    )

    if movie_data_summ.empty:
        logger.warning(f"No summary data found for movie_id: {movie_id}")
        raise HTTPException(status_code=404, detail="Movie not found")

    texts = movie_data_summ["summary"].tolist()

    logger.info(f"Extracted {len(texts)} summary texts to list")

    summary_text = summary_model.generate(texts)

    logger.info(f"Generated summary text length: {len(summary_text)}")

    valid_ratings = movie_data_rating[
        (movie_data_rating["user_rating"] > 0) & (movie_data_rating["user_rating"] <= 10)
    ]["user_rating"]

    logger.info(f"Filtered valid ratings shape: {valid_ratings.shape}")

    avg_rating = round(valid_ratings.mean(), 1) if not valid_ratings.empty else None

    logger.info(f"Calculated average rating: {avg_rating}")

    return SummaryResponse(movie_id=movie_id, summary=summary_text, average_rating=avg_rating)
