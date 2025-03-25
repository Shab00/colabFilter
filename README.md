# LastFM Music Recommendation System

## Project Description

A collaborative filtering recommendation system built with FastAI that suggests artists based on user listening history from the LastFM dataset. The system uses embedding-based similarity to find and recommend artists with similar listening patterns.

## Technical Implementation

### Core Components
- **Dot Product Model**: Custom PyTorch module with user and artist embeddings
- **Data Processing**:
  - Log transformation and standardization of scrobble counts
  - Merging of artist metadata
- **Similarity Engine**: Cosine similarity on artist embeddings
- **Training**: 15 epochs with 1-cycle policy (LR=1e-3) using MSE loss

### Key Features
- Artist-to-artist recommendations
- Combined similarity scoring for multiple input artists
- Standardized playcount normalization
- Embedding visualization capabilities

## Code Structure

```python
# Main Components:
DotProduct()          # Custom embedding model
create_artist_similarity_df()  # Generates similarity matrix
recommend_artists_by_artists() # Recommendation interface

# Data Flow:
scrobbles → log transform → standardization → embedding training → similarity matrix

# Requirements 
fastai>=2.0
pandas
numpy
scikit-learn
seaborn
torch