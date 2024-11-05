from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import pickle
import os
from typing import List
import logging
from datetime import datetime

# Initialize FastAPI app
app = FastAPI(title="Travel Recommendation API")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename=f'api_logs_{datetime.now().strftime("%Y%m%d")}.log'
)


class RecommendationResponse(BaseModel):
    user_id: str
    recommendations: List[dict]
    timestamp: str


class ModelService:
    def __init__(self):
        self.model = None
        self.df_original = None
        self.item_similarities = None

    def load_model(self, model_path: str):
        """Load the trained model"""
        try:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            logging.info("Model loaded successfully")
        except Exception as e:
            logging.error(f"Error loading model: {str(e)}")
            raise

    def load_data(self, data_path: str):
        """Load the original dataset"""
        try:
            self.df_original = pd.read_csv(data_path)
            logging.info("Data loaded successfully")
        except Exception as e:
            logging.error(f"Error loading data: {str(e)}")
            raise

    def load_similarities(self, similarities_path: str):
        """Load pre-computed item similarities"""
        try:
            with open(similarities_path, 'rb') as f:
                self.item_similarities = pickle.load(f)
            logging.info("Similarities loaded successfully")
        except Exception as e:
            logging.error(f"Error loading similarities: {str(e)}")
            raise

    def get_recommendations(self, user_id: str, n_recommendations: int = 5) -> List[dict]:
        """Generate recommendations for a user"""
        try:
            # Check if user exists in the dataset
            if user_id not in self.df_original['userID'].unique():
                raise HTTPException(status_code=404, detail=f"User {user_id} not found")

            # Get user's visited places
            visited_items = set(
                self.df_original[self.df_original['userID'] == user_id]['itemID']
            )

            # Get user's SIDO distribution
            user_sidos = (
                self.df_original[self.df_original['userID'] == user_id]['SIDO']
                .value_counts(normalize=True)
                .to_dict()
            )

            # Get all items
            all_items = set(self.df_original['itemID'].unique())

            # Get candidate items (not visited by user)
            candidate_items = list(all_items - visited_items)

            # Generate predictions for candidate items
            predictions = []
            for item in candidate_items:
                pred = self.model.predict(user_id, item)
                predictions.append((item, pred.est))

            # Sort by predicted rating
            predictions.sort(key=lambda x: x[1], reverse=True)

            # Get top N recommendations considering diversity
            recommendations = []
            selected_items = []

            for item, score in predictions:
                if len(selected_items) >= n_recommendations:
                    break

                # Calculate diversity score
                diversity_score = 1.0
                if selected_items:
                    sim_scores = []
                    for selected_item in selected_items:
                        sim = self.item_similarities.get(item, {}).get(selected_item, 0)
                        sim_scores.append(abs(sim))
                    diversity_score = 1 - (sum(sim_scores) / len(sim_scores))

                # Get item SIDO
                item_sido = self.df_original[
                    self.df_original['itemID'] == item
                    ]['SIDO'].iloc[0]

                # Get SIDO score
                sido_score = user_sidos.get(item_sido, 0.1)

                # Calculate final score
                final_score = score * 0.7 + diversity_score * 0.3
                final_score *= sido_score

                if len(recommendations) < n_recommendations:
                    recommendations.append({
                        'item_id': item,
                        'sido': item_sido,
                        'predicted_rating': float(score),
                        'diversity_score': float(diversity_score),
                        'confidence_score': float(final_score)
                    })
                    selected_items.append(item)

            return recommendations

        except Exception as e:
            logging.error(f"Error generating recommendations: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error generating recommendations: {str(e)}"
            )


# Initialize model service
model_service = ModelService()


@app.on_event("startup")
async def startup_event():
    """Load model and data on startup"""
    try:
        # Load best model
        model_service.load_model('./experiments/best_model/model.pkl')

        # Load original data
        model_service.load_data('./preprocessed/dfE.csv')

        # Load pre-computed similarities
        model_service.load_similarities('./data/item_similarities.pkl')

        logging.info("Startup completed successfully")
    except Exception as e:
        logging.error(f"Startup error: {str(e)}")
        raise


@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Travel Recommendation API"}


@app.get("/recommend/{user_id}", response_model=RecommendationResponse)
async def get_recommendations(user_id: str, n_recommendations: int = 5):
    """Get recommendations for a user"""
    try:
        recommendations = model_service.get_recommendations(
            user_id,
            n_recommendations
        )

        return {
            "user_id": user_id,
            "recommendations": recommendations,
            "timestamp": datetime.now().isoformat()
        }

    except HTTPException as e:
        raise e
    except Exception as e:
        logging.error(f"Error in recommendation endpoint: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error"
        )


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model_service.model is not None,
        "data_loaded": model_service.df_original is not None,
        "similarities_loaded": model_service.item_similarities is not None
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",  # 모든 IP에서 접근 가능
        port=8000,
        reload=True
    )
