import os
import json
import nltk
import numpy as np
import requests
import time
import re
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import random

# Ensure NLTK data is downloaded
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab')

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Constants
DATA_FILE = "movie_data.json"
API_KEY_FILE = "tmdb_api_key.txt"
TMDB_BASE_URL = "https://api.themoviedb.org/3"

# Updated target counts
HOLLYWOOD_COUNT = 3000  # Modified as requested
BOLLYWOOD_COUNT = 1000  # Modified as requested
SOUTH_INDIAN_COUNT = 500
WEB_SERIES_COUNT = 500

# Total target count
TARGET_MOVIE_COUNT = 5026

class MovieRecommender:
    def __init__(self):
        self.movies = []
        self.tfidf_matrix = None
        self.vectorizer = None
        self.api_key = self._load_api_key()
        self.unique_movie_ids = set()  # To track unique movies
        
        # Load existing data or fetch new data
        if os.path.exists(DATA_FILE):
            self._load_data()
            # If loaded data is less than target, fetch more
            if len(self.movies) < TARGET_MOVIE_COUNT:
                print(f"Only {len(self.movies)} movies in dataset, fetching more to reach {TARGET_MOVIE_COUNT}...")
                self._fetch_additional_data()
        else:
            self._fetch_and_process_data()
        
        self._prepare_tfidf()
    
    def _load_api_key(self):
        """Load TMDB API key from file"""
        try:
            with open(API_KEY_FILE, 'r') as f:
                return f.read().strip()
        except FileNotFoundError:
            print(f"Error: {API_KEY_FILE} not found. Please create this file with your TMDB API key.")
            return "YOUR_API_KEY_HERE"  # Placeholder for testing
    
    def _load_data(self):
        """Load movie data from JSON file"""
        print("Loading existing movie data...")
        with open(DATA_FILE, 'r', encoding='utf-8') as f:
            self.movies = json.load(f)
        # Populate the unique IDs set
        self.unique_movie_ids = set(movie['id'] for movie in self.movies)
        print(f"Loaded {len(self.movies)} movies")
    
    def _fetch_and_process_data(self):
        """Fetch a large dataset of movies from TMDB API using multiple methods"""
        print(f"Fetching {TARGET_MOVIE_COUNT} movies from TMDB API...")
        self.movies = []
        self.unique_movie_ids = set()
        
        # Get list of all available genres
        genres = self._get_genres()
        
        # Track progress
        start_time = time.time()
        
        # Track movie counts by category
        hollywood_count = 0
        bollywood_count = 0
        south_indian_count = 0
        web_series_count = 0
        
        # 1. Fetch Hollywood/International movies first (popular movies)
        print(f"Phase 1: Fetching Hollywood/International movies (target: {HOLLYWOOD_COUNT})...")
        
        # 1.1 Fetch popular movies
        self._fetch_from_endpoint("movie/popular", pages=20)
        hollywood_count = len(self.unique_movie_ids)
        self._save_progress("Popular Movies")
        
        # 1.2 Fetch top-rated movies
        self._fetch_from_endpoint("movie/top_rated", pages=20)
        hollywood_count = len(self.unique_movie_ids)
        self._save_progress("Top Rated Movies")
        
        # 1.3 Fetch movies by year (recent years first)
        current_year = datetime.now().year
        for year in range(current_year, current_year - 30, -1):  # 30 years back
            if hollywood_count >= HOLLYWOOD_COUNT:
                break
                
            # Make sure we're specifically getting movies from this year
            self._fetch_by_year(year, max_pages=3, strict_year=True)
            hollywood_count = len(self.unique_movie_ids)
            self._save_progress(f"Year {year}")
        
        # 1.4 Fetch movies by genre
        for genre in genres:
            if hollywood_count >= HOLLYWOOD_COUNT:
                break
            self._fetch_by_genre(genre['id'], genre['name'], max_pages=5)
            hollywood_count = len(self.unique_movie_ids)
            self._save_progress(f"Genre {genre['name']}")
        
        # 1.5 Fetch by language (focus on English)
        if hollywood_count < HOLLYWOOD_COUNT:
            self._fetch_by_language("en", max_pages=20)
            hollywood_count = len(self.unique_movie_ids)
            self._save_progress("English language movies")
        
        # 1.6 Fetch by top studios
        top_studios = [
            420,   # Marvel Studios
            2,     # Disney
            33,    # Universal Pictures
            4,     # Paramount
            174,   # Warner Bros. Pictures
            7505,  # Sony Pictures
            25,    # 20th Century Fox
            4171,  # Pixar
            41,    # Dreamworks
        ]
        
        for studio_id in top_studios:
            if hollywood_count >= HOLLYWOOD_COUNT:
                break
            self._fetch_by_company(studio_id, max_pages=5)
            hollywood_count = len(self.unique_movie_ids)
            self._save_progress(f"Studio {studio_id}")
        
        # 2. Fetch Bollywood movies (Hindi cinema)
        print(f"Phase 2: Fetching Bollywood movies (target: {BOLLYWOOD_COUNT})...")
        bollywood_start_count = len(self.unique_movie_ids)
        bollywood_fetched = 0
        
        # 2.1 Using discover endpoint with Hindi language parameter
        page = 1
        while bollywood_fetched < BOLLYWOOD_COUNT:
            url = f"{TMDB_BASE_URL}/discover/movie?api_key={self.api_key}&with_original_language=hi&page={page}&sort_by=popularity.desc"
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    data = response.json()
                    results = data.get('results', [])
                    if not results:
                        break
                        
                    before_count = len(self.unique_movie_ids)
                    self._process_movie_results(results, is_bollywood=True)
                    after_count = len(self.unique_movie_ids)
                    bollywood_fetched += (after_count - before_count)
                        
                    self._save_progress(f"Bollywood movies (page {page})")
                    page += 1
                    time.sleep(0.5)  # Prevent rate limiting
                        
                    # Break if we've reached end of results
                    if page > data.get('total_pages', 1) or page > 100:  # Allow up to 100 pages
                        break
                else:
                    print(f"Error {response.status_code} when fetching Bollywood movies")
                    time.sleep(2)
            except Exception as e:
                print(f"Exception while fetching Bollywood movies: {e}")
                time.sleep(2)
                
        print(f"Fetched {bollywood_fetched} Bollywood movies")
        
        # 2.2 Fetch by popular Bollywood studios/production companies
        if bollywood_fetched < BOLLYWOOD_COUNT:
            bollywood_studios = [
                1569,   # Yash Raj Films
                2515,   # Dharma Productions
                1913,   # Excel Entertainment
                5626,   # Red Chillies Entertainment
                1884,   # UTV Motion Pictures
                3538,   # T-Series
                7294,   # Viacom18 Studios
                128250, # Aamir Khan Productions
                156782  # Sanjay Leela Bhansali Productions
            ]
            
            for studio_id in bollywood_studios:
                if bollywood_fetched >= BOLLYWOOD_COUNT:
                    break
                    
                # Use a more targeted approach that combines company and language
                url = f"{TMDB_BASE_URL}/discover/movie?api_key={self.api_key}&with_companies={studio_id}&with_original_language=hi&page=1&sort_by=popularity.desc"
                try:
                    response = requests.get(url)
                    if response.status_code == 200:
                        data = response.json()
                        total_pages = min(data.get('total_pages', 1), 10)  # Limit to 10 pages per studio
                        
                        for page in range(1, total_pages + 1):
                            if page > 1:  # Skip first page as we already fetched it
                                url = f"{TMDB_BASE_URL}/discover/movie?api_key={self.api_key}&with_companies={studio_id}&with_original_language=hi&page={page}&sort_by=popularity.desc"
                                response = requests.get(url)
                                if response.status_code != 200:
                                    continue
                                data = response.json()
                                
                            before_count = len(self.unique_movie_ids)
                            self._process_movie_results(data.get('results', []), is_bollywood=True)
                            after_count = len(self.unique_movie_ids)
                            bollywood_fetched += (after_count - before_count)
                            
                            self._save_progress(f"Bollywood studio {studio_id} (page {page})")
                            time.sleep(0.5)
                except Exception as e:
                    print(f"Error fetching from Bollywood studio {studio_id}: {e}")
        
        # 3. Fetch South Indian movies (Tamil, Telugu, Malayalam, Kannada)
        print(f"Phase 3: Fetching South Indian movies (target: {SOUTH_INDIAN_COUNT})...")
        south_indian_fetched = 0
        south_indian_languages = {
            "ta": "Tamil", 
            "te": "Telugu", 
            "ml": "Malayalam", 
            "kn": "Kannada"
        }
                
        # 3.1 Fetch by South Indian languages
        for language_code, language_name in south_indian_languages.items():
            language_target = SOUTH_INDIAN_COUNT // len(south_indian_languages)
            language_count = 0
            page = 1
                    
            while language_count < language_target and south_indian_fetched < SOUTH_INDIAN_COUNT:
                url = f"{TMDB_BASE_URL}/discover/movie?api_key={self.api_key}&with_original_language={language_code}&page={page}&sort_by=popularity.desc"
                try:
                    response = requests.get(url)
                    if response.status_code == 200:
                        data = response.json()
                        results = data.get('results', [])
                        if not results:
                            break
                                
                        before_count = len(self.unique_movie_ids)
                        # Add tag for the language to make searching easier
                        for result in results:
                            result['language_tag'] = language_name.lower() + " south indian"
                            
                        self._process_movie_results(results, is_south_indian=True, language=language_name)
                        after_count = len(self.unique_movie_ids)
                        language_count += (after_count - before_count)
                        south_indian_fetched += (after_count - before_count)
                                
                        self._save_progress(f"{language_name} movies (page {page})")
                        page += 1
                        time.sleep(0.5)
                                
                        if page > data.get('total_pages', 1) or page > 20:  # Increased to 20 pages to get more results
                            break
                    else:
                        print(f"Error {response.status_code} when fetching {language_name} movies")
                        time.sleep(2)
                except Exception as e:
                    print(f"Exception while fetching {language_name} movies: {e}")
                    time.sleep(2)
        
        # 4. Fetch web series (TV shows)
        print(f"Phase 4: Fetching web series (target: {WEB_SERIES_COUNT})...")
        web_series_fetched = 0
        
        # 4.1 Fetch popular web series first
        page = 1
        while web_series_fetched < WEB_SERIES_COUNT:
            url = f"{TMDB_BASE_URL}/tv/popular?api_key={self.api_key}&page={page}"
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    data = response.json()
                    results = data.get('results', [])
                    if not results:
                        break
                    
                    # Process TV shows similar to movies
                    before_count = len(self.unique_movie_ids)
                    for show in results:
                        if show.get('id') and show['id'] not in self.unique_movie_ids:
                            try:
                                show_details = self._get_tv_details(show['id'])
                                if show_details:
                                    # Add 'web series' tag for easier searching
                                    show_details['document'] += " web series tv show"
                                    self.movies.append(show_details)
                                    self.unique_movie_ids.add(show['id'])
                                    time.sleep(0.1)  # Prevent rate limiting
                            except Exception as e:
                                print(f"Error processing TV show {show.get('id')}: {e}")
                    
                    after_count = len(self.unique_movie_ids)
                    web_series_fetched += (after_count - before_count)
                    
                    self._save_progress(f"Popular web series (page {page})")
                    page += 1
                    time.sleep(0.5)
                    
                    if page > data.get('total_pages', 1) or page > 15:  # Limit to 15 pages
                        break
                else:
                    print(f"Error {response.status_code} when fetching popular web series")
                    time.sleep(2)
            except Exception as e:
                print(f"Exception while fetching popular web series: {e}")
                time.sleep(2)
        
        # Final save
        with open(DATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(self.movies, f, ensure_ascii=False, indent=2)
            
        elapsed_time = (time.time() - start_time) / 60
        
        # Print summary
        print("\n=== FINAL SUMMARY ===")
        print(f"Total movies/shows in dataset: {len(self.movies)}")
        print(f"- Hollywood/International: {hollywood_count}")
        print(f"- Bollywood: {bollywood_fetched}")
        print(f"- South Indian: {south_indian_fetched}")
        print(f"- Web Series: {web_series_fetched}")
        print(f"Fetched and saved in {elapsed_time:.2f} minutes")
    
    def _fetch_additional_data(self):
        """Fetch additional movies to reach the target count"""
        current_count = len(self.movies)
        if current_count >= TARGET_MOVIE_COUNT:
            return
            
        # How many more movies we need
        needed = TARGET_MOVIE_COUNT - current_count
        
        # Count movies by category in current dataset
        hollywood_count = 0
        bollywood_count = 0
        south_indian_count = 0
        web_series_count = 0
        
        for movie in self.movies:
            content_type = movie.get('content_type', 'movie')
            language = movie.get('language', 'unknown')
            
            if content_type == 'tv':
                web_series_count += 1
            elif language == 'hi':
                bollywood_count += 1
            elif language in ['ta', 'te', 'ml', 'kn']:
                south_indian_count += 1
            else:
                hollywood_count += 1
        
        print(f"\nCurrent counts:")
        print(f"- Hollywood/International: {hollywood_count}")
        print(f"- Bollywood: {bollywood_count}")
        print(f"- South Indian: {south_indian_count}")
        print(f"- Web Series: {web_series_count}")
        
        # Calculate how many more of each category we need
        need_hollywood = max(0, HOLLYWOOD_COUNT - hollywood_count)
        need_bollywood = max(0, BOLLYWOOD_COUNT - bollywood_count)
        need_south_indian = max(0, SOUTH_INDIAN_COUNT - south_indian_count)
        need_web_series = max(0, WEB_SERIES_COUNT - web_series_count)
        
        print(f"\nNeed to fetch:")
        print(f"- Hollywood/International: {need_hollywood}")
        print(f"- Bollywood: {need_bollywood}")
        print(f"- South Indian: {need_south_indian}")
        print(f"- Web Series: {need_web_series}")
        
        # First try to fetch more Bollywood movies if needed
        if need_bollywood > 0:
            print(f"Fetching {need_bollywood} more Bollywood movies...")
            # Using Hindi language filter
            self._fetch_by_language("hi", max_pages=min(50, need_bollywood // 20 + 1))
        
        # Then fetch more South Indian movies if needed
        if need_south_indian > 0:
            print(f"Fetching {need_south_indian} more South Indian movies...")
            languages = ["ta", "te", "ml", "kn"]
            for lang in languages:
                self._fetch_by_language(lang, max_pages=min(20, need_south_indian // 20 + 1))
        
        # Then fetch more web series if needed
        if need_web_series > 0:
            print(f"Fetching {need_web_series} more web series...")
            pages = min(20, need_web_series // 20 + 1)
            
            # Popular TV shows
            url = f"{TMDB_BASE_URL}/tv/popular?api_key={self.api_key}&page=1"
            for page in range(1, pages + 1):
                    response = requests.get(url.replace("page=1", f"page={page}"))
                    if response.status_code == 200:
                        data = response.json()
                        for show in data.get('results', []):
                            if show.get('id') and show['id'] not in self.unique_movie_ids:
                                try:
                                    show_details = self._get_tv_details(show['id'])
                                    if show_details:
                                        show_details['document'] += " web series tv show"
                                        self.movies.append(show_details)
                                        self.unique_movie_ids.add(show['id'])
                                        time.sleep(0.1)
                                except Exception as e:
                                    print(f"Error processing TV show {show.get('id')}: {e}")
        
        # Finally, fetch more Hollywood movies if needed
        if need_hollywood > 0:
            print(f"Fetching {need_hollywood} more Hollywood/International movies...")
            
            # Try to fetch by years we might not have covered yet
            years_to_try = list(range(1980, datetime.now().year))
            random.shuffle(years_to_try)  # Randomize to get variety
            
            for year in years_to_try:
                if len(self.unique_movie_ids) >= TARGET_MOVIE_COUNT:
                    break
                self._fetch_by_year(year, max_pages=2, strict_year=True)
        
        # Save final dataset
        with open(DATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(self.movies, f, ensure_ascii=False, indent=2)
            
        print(f"Dataset updated to {len(self.movies)} movies/shows")
    
    # The rest of the methods remain largely the same
    
    def _get_genres(self):
        """Get list of all available movie genres from TMDB"""
        url = f"{TMDB_BASE_URL}/genre/movie/list?api_key={self.api_key}"
        try:
            response = requests.get(url)
            if response.status_code == 200:
                return response.json().get('genres', [])
            else:
                print(f"Error {response.status_code} when fetching genres")
                return []
        except Exception as e:
            print(f"Exception while fetching genres: {e}")
            return []
    
    def _fetch_from_endpoint(self, endpoint, pages=10):
        """Fetch movies from a specific TMDB endpoint"""
        for page in range(1, pages + 1):
            url = f"{TMDB_BASE_URL}/{endpoint}?api_key={self.api_key}&page={page}"
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    data = response.json()
                    self._process_movie_results(data.get('results', []))
                    print(f"Fetched {endpoint} page {page}/{pages}")
                    time.sleep(0.5)  # Prevent rate limiting
                else:
                    print(f"Error {response.status_code} when fetching {endpoint} page {page}")
                    time.sleep(2)
            except Exception as e:
                print(f"Exception while fetching {endpoint} page {page}: {e}")
                time.sleep(2)
    
    def _fetch_by_year(self, year, max_pages=5, strict_year=False):
        """Fetch movies released in a specific year"""
        for page in range(1, max_pages + 1):
            # For strict year matching, use both primary_release_year and year parameters
            if strict_year:
                url = f"{TMDB_BASE_URL}/discover/movie?api_key={self.api_key}&primary_release_year={year}&year={year}&page={page}&sort_by=popularity.desc"
            else:
                url = f"{TMDB_BASE_URL}/discover/movie?api_key={self.api_key}&primary_release_year={year}&page={page}&sort_by=popularity.desc"
            
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    data = response.json()
                    results = data.get('results', [])
                    
                    # Additional verification for strict year matching
                    if strict_year:
                        filtered_results = []
                        for movie in results:
                            release_date = movie.get('release_date', '')
                            if release_date and release_date.startswith(str(year)):
                                filtered_results.append(movie)
                        results = filtered_results
                    
                    self._process_movie_results(results)
                    print(f"Fetched year {year} page {page}/{max_pages}")
                    time.sleep(0.5)
                else:
                    print(f"Error {response.status_code} when fetching year {year} page {page}")
                    time.sleep(2)
            except Exception as e:
                print(f"Exception while fetching year {year} page {page}: {e}")
                time.sleep(2)
    
    def _fetch_by_genre(self, genre_id, genre_name, max_pages=5):
        """Fetch movies by genre"""
        for page in range(1, max_pages + 1):
            url = f"{TMDB_BASE_URL}/discover/movie?api_key={self.api_key}&with_genres={genre_id}&page={page}&sort_by=popularity.desc"
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    data = response.json()
                    results = data.get('results', [])
                    
                    # Add genre tag for easier searching
                    for movie in results:
                        movie['genre_tag'] = genre_name.lower()
                        
                    self._process_movie_results(results)
                    print(f"Fetched genre {genre_name} page {page}/{max_pages}")
                    time.sleep(0.5)
                else:
                    print(f"Error {response.status_code} when fetching genre {genre_name} page {page}")
                    time.sleep(2)
            except Exception as e:
                print(f"Exception while fetching genre {genre_name} page {page}: {e}")
                time.sleep(2)
    
    def _fetch_by_language(self, language_code, max_pages=10):
        """Fetch movies by original language"""
        for page in range(1, max_pages + 1):
            url = f"{TMDB_BASE_URL}/discover/movie?api_key={self.api_key}&with_original_language={language_code}&page={page}&sort_by=popularity.desc"
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    data = response.json()
                    self._process_movie_results(data.get('results', []))
                    print(f"Fetched language {language_code} page {page}/{max_pages}")
                    time.sleep(0.5)
                else:
                    print(f"Error {response.status_code} when fetching language {language_code} page {page}")
                    time.sleep(2)
            except Exception as e:
                print(f"Exception while fetching language {language_code} page {page}: {e}")
                time.sleep(2)
    
    def _fetch_by_company(self, company_id, max_pages=5):
        """Fetch movies by production company"""
        for page in range(1, max_pages + 1):
            url = f"{TMDB_BASE_URL}/discover/movie?api_key={self.api_key}&with_companies={company_id}&page={page}&sort_by=popularity.desc"
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    data = response.json()
                    self._process_movie_results(data.get('results', []))
                    print(f"Fetched company {company_id} page {page}/{max_pages}")
                    time.sleep(0.5)
                else:
                    print(f"Error {response.status_code} when fetching company {company_id} page {page}")
                    time.sleep(2)
            except Exception as e:
                print(f"Exception while fetching company {company_id} page {page}: {e}")
                time.sleep(2)
    
    def _process_movie_results(self, results, is_bollywood=False, is_south_indian=False, language=None):
        """Process movie results and add to dataset if not already present"""
        count = 0
        for movie in results:
            movie_id = movie.get('id')
            if movie_id and movie_id not in self.unique_movie_ids:
                try:
                    # Get additional movie details
                    movie_details = self._get_movie_details(movie_id)
                    
                    if movie_details:
                        # Add specific tags based on movie type
                        if is_bollywood:
                            movie_details['document'] += " bollywood hindi indian"
                        elif is_south_indian:
                            movie_details['document'] += f" {language.lower()} south indian"
                        
                        # Add any genre or language tags that were added during fetching
                        if 'genre_tag' in movie:
                            movie_details['document'] += f" {movie['genre_tag']}"
                        if 'language_tag' in movie:
                            movie_details['document'] += f" {movie['language_tag']}"
                        
                        self.movies.append(movie_details)
                        self.unique_movie_ids.add(movie_id)
                        count += 1
                except Exception as e:
                    print(f"Error processing movie {movie_id}: {e}")
        
        print(f"Added {count} new movies from batch")
    
    def _get_movie_details(self, movie_id, prefer_hindi=False):
        """Get detailed information about a specific movie"""
        url = f"{TMDB_BASE_URL}/movie/{movie_id}?api_key={self.api_key}&append_to_response=credits,keywords"
        try:
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                
                # Basic movie information
                title = data.get('title', '')
                original_title = data.get('original_title', '')
                overview = data.get('overview', '')
                release_date = data.get('release_date', '')
                
                # Language handling - for Bollywood preferences
                original_language = data.get('original_language', '')
                if prefer_hindi and original_language != 'hi':
                    return None
                
                # Get genres, cast, crew
                genres = [genre['name'] for genre in data.get('genres', [])]
                
                # Get director and top cast
                director = ""
                cast = []
                
                credits = data.get('credits', {})
                crew = credits.get('crew', [])
                actors = credits.get('cast', [])
                
                for person in crew:
                    if person.get('job') == 'Director':
                        director = person.get('name', '')
                        break
                
                for actor in actors[:10]:  # Get top 10 cast
                    if actor.get('name'):
                        cast.append(actor.get('name'))
                
                # Get keywords/tags
                keywords = []
                if 'keywords' in data and 'keywords' in data['keywords']:
                    keywords = [kw['name'] for kw in data['keywords']['keywords']]
                
                # Create a comprehensive document for text search
                document = f"{title} {original_title} {overview} "
                document += f"{' '.join(genres)} {director} {' '.join(cast)} {' '.join(keywords)} "
                document += f"{release_date[:4] if release_date else ''} "  # Add year for searching by year
                
                # Add movie or specific category identifiers
                document += "movie film "
                
               # Add language specific identifiers
                if original_language == 'en':
                    document += "english "
                elif original_language == 'hi':
                    document += "hindi bollywood "
                elif original_language == 'ta':
                    document += "tamil kollywood "
                elif original_language == 'te':
                    document += "telugu tollywood "
                elif original_language == 'ml':
                    document += "malayalam mollywood "
                elif original_language == 'kn':
                    document += "kannada sandalwood "
                
                # Return structured movie data
                return {
                    'id': movie_id,
                    'title': title,
                    'original_title': original_title,
                    'overview': overview,
                    'release_date': release_date,
                    'genres': genres,
                    'director': director,
                    'cast': cast,
                    'keywords': keywords,
                    'language': original_language,
                    'document': document,
                    'content_type': 'movie',
                    'poster_path': data.get('poster_path', ''),
                    'backdrop_path': data.get('backdrop_path', ''),
                    'popularity': data.get('popularity', 0),
                    'vote_average': data.get('vote_average', 0)
                }
            else:
                print(f"Error {response.status_code} when fetching movie details for ID {movie_id}")
                return None
        except Exception as e:
            print(f"Exception while fetching movie details for ID {movie_id}: {e}")
            return None
    
    def _get_tv_details(self, show_id):
        """Get detailed information about a specific TV show"""
        url = f"{TMDB_BASE_URL}/tv/{show_id}?api_key={self.api_key}&append_to_response=credits,keywords"
        try:
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                
                # Basic show information
                title = data.get('name', '')
                original_title = data.get('original_name', '')
                overview = data.get('overview', '')
                first_air_date = data.get('first_air_date', '')
                
                # Get genres, cast, crew
                genres = [genre['name'] for genre in data.get('genres', [])]
                
                # Get creator and top cast
                creators = []
                cast = []
                
                for person in data.get('created_by', []):
                    if person.get('name'):
                        creators.append(person.get('name'))
                
                credits = data.get('credits', {})
                actors = credits.get('cast', [])
                
                for actor in actors[:10]:  # Get top 10 cast
                    if actor.get('name'):
                        cast.append(actor.get('name'))
                
                # Get keywords/tags
                keywords = []
                if 'keywords' in data and 'results' in data['keywords']:
                    keywords = [kw['name'] for kw in data['keywords']['results']]
                
                # Create a comprehensive document for text search
                document = f"{title} {original_title} {overview} "
                document += f"{' '.join(genres)} {' '.join(creators)} {' '.join(cast)} {' '.join(keywords)} "
                document += f"{first_air_date[:4] if first_air_date else ''} "  # Add year for searching by year
                
                # Add TV show specific identifiers
                document += "tv television series show web series "
                
                # Add language specific identifiers
                original_language = data.get('original_language', '')
                if original_language == 'en':
                    document += "english "
                elif original_language == 'hi':
                    document += "hindi "
                
                # Return structured TV show data
                return {
                    'id': show_id,
                    'title': title,
                    'original_title': original_title,
                    'overview': overview,
                    'release_date': first_air_date,
                    'genres': genres,
                    'creators': creators,
                    'cast': cast,
                    'keywords': keywords,
                    'language': original_language,
                    'document': document,
                    'content_type': 'tv',
                    'poster_path': data.get('poster_path', ''),
                    'backdrop_path': data.get('backdrop_path', ''),
                    'popularity': data.get('popularity', 0),
                    'vote_average': data.get('vote_average', 0),
                    'number_of_seasons': data.get('number_of_seasons', 0),
                    'number_of_episodes': data.get('number_of_episodes', 0)
                }
            else:
                print(f"Error {response.status_code} when fetching TV show details for ID {show_id}")
                return None
        except Exception as e:
            print(f"Exception while fetching TV show details for ID {show_id}: {e}")
            return None
    
    def _save_progress(self, description):
        """Save the current progress"""
        print(f"Progress: {len(self.unique_movie_ids)} movies - {description}")
        # Save every 100 movies to avoid losing data if process is interrupted
        if len(self.unique_movie_ids) % 100 == 0:
            with open(DATA_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.movies, f, ensure_ascii=False, indent=2)
            print(f"Saved progress at {len(self.unique_movie_ids)} movies")
    
    def _prepare_tfidf(self):
        """Prepare TF-IDF matrix for movie similarity"""
        print("Preparing TF-IDF matrix for recommendations...")
        
        # Extract documents for vectorization
        documents = [self._preprocess_text(movie['document']) for movie in self.movies]
        
        # Create TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2)  # Use both unigrams and bigrams
        )
        
        # Create TF-IDF matrix
        self.tfidf_matrix = self.vectorizer.fit_transform(documents)
        print(f"TF-IDF matrix shape: {self.tfidf_matrix.shape}")
    
    def _preprocess_text(self, text):
        """Preprocess text for TF-IDF"""
        if not text:
            return ""
            
        # Lowercase
        text = text.lower()
        
        # Remove punctuation and special characters
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]
        
        # Lemmatize
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
        
        return ' '.join(tokens)
    
    def search(self, query, top_n=10):
        """Search for movies based on text query"""
        # Preprocess query
        processed_query = self._preprocess_text(query)
        
        # Transform query to TF-IDF vector
        query_vector = self.vectorizer.transform([processed_query])
        
        # Calculate similarity
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        # Get indices of top similar movies
        top_indices = similarities.argsort()[:-top_n-1:-1]
        
        # Get top movies
        top_movies = [self.movies[i] for i in top_indices]
        
        # Add similarity scores
        for i, movie in enumerate(top_movies):
            movie['similarity'] = float(similarities[top_indices[i]])
        
        return top_movies
    
    def get_recommendations(self, movie_id, top_n=10):
        """Get movie recommendations based on a specific movie"""
        # Find the movie in our dataset
        movie_index = None
        for i, movie in enumerate(self.movies):
            if movie['id'] == movie_id:
                movie_index = i
                break
        
        if movie_index is None:
            return []
        
        # Calculate similarity with all other movies
        movie_vector = self.tfidf_matrix[movie_index]
        similarities = cosine_similarity(movie_vector, self.tfidf_matrix).flatten()
        
        # Get indices of top similar movies (excluding the movie itself)
        indices = list(range(len(similarities)))
        indices.sort(key=lambda x: similarities[x], reverse=True)
        indices = [i for i in indices if i != movie_index][:top_n]
        
        # Get top similar movies
        similar_movies = [self.movies[i] for i in indices]
        
        # Add similarity scores
        for i, movie in enumerate(similar_movies):
            movie['similarity'] = float(similarities[indices[i]])
        
        return similar_movies
    
    def get_random_recommendations(self, top_n=10):
        """Get random movie recommendations"""
        # Get random indices
        indices = random.sample(range(len(self.movies)), min(top_n, len(self.movies)))
        
        # Get random movies
        random_movies = [self.movies[i] for i in indices]
        
        return random_movies
    
    def get_popular_recommendations(self, top_n=10):
        """Get popular movie recommendations"""
        # Sort movies by popularity
        sorted_movies = sorted(self.movies, key=lambda x: x.get('popularity', 0), reverse=True)
        
        # Get top popular movies
        popular_movies = sorted_movies[:top_n]
        
        return popular_movies
    
    def get_top_rated_recommendations(self, top_n=10):
        """Get top rated movie recommendations"""
        # Filter movies with at least some votes
        voted_movies = [movie for movie in self.movies if movie.get('vote_average', 0) > 0]
        
        # Sort movies by rating
        sorted_movies = sorted(voted_movies, key=lambda x: x.get('vote_average', 0), reverse=True)
        
        # Get top rated movies
        top_rated_movies = sorted_movies[:top_n]
        
        return top_rated_movies
    
    def get_movie_details(self, movie_id):
        """Get details for a specific movie by ID"""
        for movie in self.movies:
            if movie['id'] == movie_id:
                return movie
        return None


# Initialize Flask application
@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/api/search', methods=['GET'])
def search_movies():
    """API endpoint for searching movies"""
    query = request.args.get('q', '')
    top_n = int(request.args.get('n', 10))
    
    if not query:
        return jsonify({'error': 'Query parameter required'}), 400
    
    recommender = MovieRecommender()
    results = recommender.search(query, top_n=top_n)
    
    return jsonify({'results': results})

@app.route('/api/recommendations', methods=['GET'])
def get_recommendations():
    """API endpoint for getting recommendations for a specific movie"""
    movie_id = request.args.get('id')
    top_n = int(request.args.get('n', 10))
    
    if not movie_id:
        return jsonify({'error': 'Movie ID parameter required'}), 400
    
    try:
        movie_id = int(movie_id)
    except ValueError:
        return jsonify({'error': 'Invalid movie ID format'}), 400
    
    recommender = MovieRecommender()
    results = recommender.get_recommendations(movie_id, top_n=top_n)
    
    return jsonify({'results': results})

@app.route('/api/movie/<int:movie_id>', methods=['GET'])
def get_movie(movie_id):
    """API endpoint for getting details of a specific movie"""
    recommender = MovieRecommender()
    movie = recommender.get_movie_details(movie_id)
    
    if movie:
        return jsonify({'movie': movie})
    else:
        return jsonify({'error': 'Movie not found'}), 404

@app.route('/api/random', methods=['GET'])
def get_random():
    """API endpoint for getting random movie recommendations"""
    top_n = int(request.args.get('n', 10))
    
    recommender = MovieRecommender()
    results = recommender.get_random_recommendations(top_n=top_n)
    
    return jsonify({'results': results})

@app.route('/api/popular', methods=['GET'])
def get_popular():
    """API endpoint for getting popular movie recommendations"""
    top_n = int(request.args.get('n', 10))
    
    recommender = MovieRecommender()
    results = recommender.get_popular_recommendations(top_n=top_n)
    
    return jsonify({'results': results})

@app.route('/api/top-rated', methods=['GET'])
def get_top_rated():
    """API endpoint for getting top rated movie recommendations"""
    top_n = int(request.args.get('n', 10))
    
    recommender = MovieRecommender()
    results = recommender.get_top_rated_recommendations(top_n=top_n)
    
    return jsonify({'results': results})


if __name__ == '__main__':
    # Initialize recommender to ensure data is loaded before serving requests
    recommender = MovieRecommender()
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)