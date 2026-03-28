# Foodie Bookmark - SE481 Information Retrieval Term Project

## Overview
Foodie Bookmark is a comprehensive food recommendation and bookmarking web application developed for the SE481 Introduction to Information Retrieval course at Chiang Mai University. It leverages robust IR techniques to help users securely discover, manage, and receive personalized recommendations for cooking recipes.

## Core Features
* **User Authentication:** Secure registration, login, and logout functionalities.
* **Advanced Recipe Search:** A unified search engine processing dish names, ingredients, and instructions. It utilizes **Field Weighting (Name: 60%, Ingredients: 30%, Instructions: 10%)** combined with TF-IDF Vectorization and Cosine Similarity to significantly enhance retrieval precision and ranking accuracy.
* **Spell Correction:** Automatic typo detection and user-friendly correction suggestions using the `spellchecker` library.
* **Instant Image Loading:** Fully implemented server-side **Image Proxy & Caching System** that optimizes (resizes and compresses) external dataset images for instant loading, preventing broken links.
* **Folder Management & Bookmarking:** Users can create custom folders, save recipes, and rate them (1-5 stars).
* **Ranked Bookmark Viewing:** A dedicated dashboard displaying all saved recipes across all folders, ranked dynamically by the user's ratings.
* **Personalized Landing Page:** Dynamically generated recipe lists including:
  1. A summary of recently saved recipes from all folders.
  2. A selection of recipes from a randomly chosen user folder.
  3. Completely random dishes for discovery.
* **Profile-Based Folder Recommendations (Explicit Relevance Feedback):** Generates personalized recommendations by creating a "Weighted Average Profile Vector". It utilizes the user's explicit 1-5 star ratings as weights against the TF-IDF vectors of saved recipes. This ensures that 5-star recipes heavily influence the recommended results, bypassing simple kNN approaches.

## Exciting IR Features (Bonus)
To further enhance the Information Retrieval experience, this application implements these additional advanced features:
1. **Content-Based "More Like This":** Displays top similar recipes inside the recipe details modal by performing a vectorized Cosine Similarity calculation against the entire dataset.
2. **Search Autocomplete:** A real-time API endpoint (`/autocomplete`) that suggests recipe names using prefix matching as the user types.
3. **Faceted Search (Time Filtering):** Allows users to filter TF-IDF search results strictly by preparation time (e.g., "Under 30 Mins") using structured metadata filtering.
4. **On-Demand User Reviews:** Dynamically retrieves and displays top user reviews (from a separate dataset) only for the highly-ranked search results to provide social proof and improve the decision-making process.

## Setup & Installation

**1. Clone the repository**
```bash
git clone [https://github.com/ninlawan-numnim/IRProject.git](https://github.com/ninlawan-numnim/IRProject.git)
cd your folder
