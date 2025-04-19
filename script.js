// DOM elements
const searchForm = document.getElementById('search-form');
const searchInput = document.getElementById('search-input');
const resultsContainer = document.getElementById('results-container');
const loadingIndicator = document.getElementById('loading-indicator');
const categoryButtons = document.querySelectorAll('.category-btn');

// API endpoint base URL - update this if your API is hosted elsewhere
const API_BASE_URL = 'http://localhost:5000/api';

// Event listeners
document.addEventListener('DOMContentLoaded', () => {
  // Show popular movies on page load
  fetchAndDisplayMovies('popular');
  
  // Set up search form submission
  if (searchForm) {
    searchForm.addEventListener('submit', handleSearch);
  }
  
  // Set up category button clicks
  categoryButtons.forEach(button => {
    button.addEventListener('click', (e) => {
      const category = e.target.dataset.category;
      fetchAndDisplayMovies(category);
      
      // Update active button styling
      categoryButtons.forEach(btn => btn.classList.remove('active'));
      e.target.classList.add('active');
    });
  });
});

// Handle search form submission
function handleSearch(e) {
  e.preventDefault();
  const query = searchInput.value.trim();
  
  if (query) {
    fetchAndDisplayMovies('search', query);
  }
}

// Fetch movies from API and display them
async function fetchAndDisplayMovies(endpoint, query = null) {
  showLoading(true);
  
  try {
    let url;
    
    switch (endpoint) {
      case 'search':
        url = `${API_BASE_URL}/search?q=${encodeURIComponent(query)}&n=20`;
        break;
      case 'popular':
        url = `${API_BASE_URL}/popular?n=20`;
        break;
      case 'top-rated':
        url = `${API_BASE_URL}/top-rated?n=20`;
        break;
      case 'random':
        url = `${API_BASE_URL}/random?n=20`;
        break;
      default:
        url = `${API_BASE_URL}/popular?n=20`;
    }
    
    const response = await fetch(url);
    const data = await response.json();
    
    if (data.error) {
      displayError(data.error);
    } else {
      displayResults(data.results, endpoint, query);
    }
  } catch (error) {
    console.error('Error fetching data:', error);
    displayError('Failed to fetch results. Please try again later.');
  } finally {
    showLoading(false);
  }
}

// Display movie results
function displayResults(movies, endpoint, query = null) {
  // Clear previous results
  resultsContainer.innerHTML = '';
  
  // Show heading based on the endpoint
  const headingText = getHeadingText(endpoint, query, movies.length);
  const heading = document.createElement('h2');
  heading.className = 'results-heading';
  heading.textContent = headingText;
  resultsContainer.appendChild(heading);
  
  // Create movie grid
  const movieGrid = document.createElement('div');
  movieGrid.className = 'movie-grid';
  
  // Add movies to grid
  movies.forEach(movie => {
    const movieCard = createMovieCard(movie);
    movieGrid.appendChild(movieCard);
  });
  
  resultsContainer.appendChild(movieGrid);
  
  // If no results, show message
  if (movies.length === 0) {
    const noResults = document.createElement('p');
    noResults.className = 'no-results';
    noResults.textContent = 'No movies found matching your criteria.';
    resultsContainer.appendChild(noResults);
  }
}

// Create a card element for a movie
function createMovieCard(movie) {
  const card = document.createElement('div');
  card.className = 'movie-card';
  card.dataset.id = movie.id;
  
  // Add poster image
  const posterUrl = movie.poster_path 
    ? `https://image.tmdb.org/t/p/w300${movie.poster_path}`
    : '/static/placeholder.jpg'; // Fallback image
  
  const posterImg = document.createElement('img');
  posterImg.className = 'movie-poster';
  posterImg.src = posterUrl;
  posterImg.alt = `${movie.title} poster`;
  posterImg.onerror = () => { posterImg.src = '/static/placeholder.jpg'; };
  
  // Create info div
  const infoDiv = document.createElement('div');
  infoDiv.className = 'movie-info';
  
  // Add title
  const title = document.createElement('h3');
  title.className = 'movie-title';
  title.textContent = movie.title;
  
  // Add year
  const year = document.createElement('p');
  year.className = 'movie-year';
  year.textContent = movie.release_date ? movie.release_date.substring(0, 4) : 'N/A';
  
  // Add rating if available
  if (movie.vote_average) {
    const rating = document.createElement('div');
    rating.className = 'movie-rating';
    const ratingValue = parseFloat(movie.vote_average).toFixed(1);
    rating.innerHTML = `<span class="star">★</span> ${ratingValue}`;
    infoDiv.appendChild(rating);
  }
  
  // Add genres
  if (movie.genres && movie.genres.length > 0) {
    const genres = document.createElement('p');
    genres.className = 'movie-genres';
    genres.textContent = movie.genres.slice(0, 3).join(', ');
    infoDiv.appendChild(genres);
  }
  
  // Add "View Details" button
  const detailsBtn = document.createElement('button');
  detailsBtn.className = 'details-btn';
  detailsBtn.textContent = 'View Details';
  detailsBtn.addEventListener('click', () => showMovieDetails(movie.id));
  
  // Assemble card
  infoDiv.prepend(year);
  infoDiv.prepend(title);
  card.appendChild(posterImg);
  card.appendChild(infoDiv);
  card.appendChild(detailsBtn);
  
  // Make the card clickable to show details
  card.addEventListener('click', (e) => {
    // Only trigger if the click wasn't on the button (which has its own handler)
    if (!e.target.classList.contains('details-btn')) {
      showMovieDetails(movie.id);
    }
  });
  
  return card;
}

// Show movie details
async function showMovieDetails(movieId) {
  showLoading(true);
  
  try {
    // Fetch detailed movie information
    const response = await fetch(`${API_BASE_URL}/movie/${movieId}`);
    const data = await response.json();
    
    if (data.error) {
      displayError(data.error);
      return;
    }
    
    const movie = data.movie;
    
    // Create modal for movie details
    const modal = document.createElement('div');
    modal.className = 'movie-modal';
    
    // Modal content
    const modalContent = document.createElement('div');
    modalContent.className = 'modal-content';
    
    // Close button
    const closeBtn = document.createElement('span');
    closeBtn.className = 'close-btn';
    closeBtn.innerHTML = '&times;';
    closeBtn.addEventListener('click', () => {
      document.body.removeChild(modal);
    });
    
    // Movie details layout (flex container)
    const detailsContainer = document.createElement('div');
    detailsContainer.className = 'details-container';
    
    // Poster
    const posterUrl = movie.poster_path 
      ? `https://image.tmdb.org/t/p/w500${movie.poster_path}`
      : '/static/placeholder.jpg';
      
    const poster = document.createElement('img');
    poster.className = 'detail-poster';
    poster.src = posterUrl;
    poster.alt = `${movie.title} poster`;
    
    // Info section
    const infoSection = document.createElement('div');
    infoSection.className = 'detail-info';
    
    // Title with year
    const titleYear = document.createElement('h2');
    titleYear.innerHTML = `${movie.title} <span>(${movie.release_date ? movie.release_date.substring(0, 4) : 'N/A'})</span>`;
    
    // Original title if different
    let originalTitleElement = '';
    if (movie.original_title && movie.original_title !== movie.title) {
      originalTitleElement = document.createElement('p');
      originalTitleElement.className = 'original-title';
      originalTitleElement.textContent = `Original title: ${movie.original_title}`;
    }
    
    // Rating and genres row
    const metaRow = document.createElement('div');
    metaRow.className = 'meta-row';
    
    if (movie.vote_average) {
      const rating = document.createElement('div');
      rating.className = 'detail-rating';
      const ratingValue = parseFloat(movie.vote_average).toFixed(1);
      rating.innerHTML = `<span class="star">★</span> ${ratingValue}`;
      metaRow.appendChild(rating);
    }
    
    if (movie.genres && movie.genres.length > 0) {
      const genres = document.createElement('div');
      genres.className = 'detail-genres';
      genres.textContent = movie.genres.join(', ');
      metaRow.appendChild(genres);
    }
    
    // Overview
    const overview = document.createElement('div');
    overview.className = 'movie-overview';
    
    const overviewHeading = document.createElement('h3');
    overviewHeading.textContent = 'Overview';
    
    const overviewText = document.createElement('p');
    overviewText.textContent = movie.overview || 'No overview available.';
    
    overview.appendChild(overviewHeading);
    overview.appendChild(overviewText);
    
    // Cast and crew
    const credits = document.createElement('div');
    credits.className = 'movie-credits';
    
    // Director
    if (movie.director) {
      const director = document.createElement('p');
      director.innerHTML = `<strong>Director:</strong> ${movie.director}`;
      credits.appendChild(director);
    }
    
    // Cast
    if (movie.cast && movie.cast.length > 0) {
      const cast = document.createElement('p');
      cast.innerHTML = `<strong>Cast:</strong> ${movie.cast.slice(0, 5).join(', ')}`;
      credits.appendChild(cast);
    }
    
    // Additional info for TV shows
    if (movie.content_type === 'tv') {
      const tvInfo = document.createElement('div');
      tvInfo.className = 'tv-info';
      
      if (movie.creators && movie.creators.length > 0) {
        const creators = document.createElement('p');
        creators.innerHTML = `<strong>Created by:</strong> ${movie.creators.join(', ')}`;
        tvInfo.appendChild(creators);
      }
      
      const episodes = document.createElement('p');
      episodes.innerHTML = `<strong>Episodes:</strong> ${movie.number_of_episodes || 'N/A'}`;
      tvInfo.appendChild(episodes);
      
      const seasons = document.createElement('p');
      seasons.innerHTML = `<strong>Seasons:</strong> ${movie.number_of_seasons || 'N/A'}`;
      tvInfo.appendChild(seasons);
      
      credits.appendChild(tvInfo);
    }
    
    // Get similar recommendations
    const recHeading = document.createElement('h3');
    recHeading.className = 'rec-heading';
    recHeading.textContent = 'Similar Recommendations';
    
    const recContainer = document.createElement('div');
    recContainer.className = 'recommendations-container';
    recContainer.innerHTML = '<p>Loading recommendations...</p>';
    
    // Fetch similar recommendations
    fetchSimilarRecommendations(movie.id, recContainer);
    
    // Assemble info section
    infoSection.appendChild(titleYear);
    if (originalTitleElement) {
      infoSection.appendChild(originalTitleElement);
    }
    infoSection.appendChild(metaRow);
    infoSection.appendChild(overview);
    infoSection.appendChild(credits);
    
    // Assemble details container
    detailsContainer.appendChild(poster);
    detailsContainer.appendChild(infoSection);
    
    // Assemble modal content
    modalContent.appendChild(closeBtn);
    modalContent.appendChild(detailsContainer);
    modalContent.appendChild(recHeading);
    modalContent.appendChild(recContainer);
    
    // Add modal to page
    modal.appendChild(modalContent);
    document.body.appendChild(modal);
    
    // Close modal when clicking outside
    modal.addEventListener('click', (e) => {
      if (e.target === modal) {
        document.body.removeChild(modal);
      }
    });
    
  } catch (error) {
    console.error('Error fetching movie details:', error);
    displayError('Failed to load movie details. Please try again later.');
  } finally {
    showLoading(false);
  }
}

// Fetch similar recommendations
async function fetchSimilarRecommendations(movieId, container) {
  try {
    const response = await fetch(`${API_BASE_URL}/recommendations?id=${movieId}&n=6`);
    const data = await response.json();
    
    if (data.error || !data.results || data.results.length === 0) {
      container.innerHTML = '<p>No similar recommendations found.</p>';
      return;
    }
    
    // Create recommendation cards
    container.innerHTML = '';
    const recGrid = document.createElement('div');
    recGrid.className = 'rec-grid';
    
    data.results.forEach(movie => {
      const recCard = document.createElement('div');
      recCard.className = 'rec-card';
      recCard.dataset.id = movie.id;
      
      const posterUrl = movie.poster_path 
        ? `https://image.tmdb.org/t/p/w200${movie.poster_path}`
        : '/static/placeholder.jpg';
        
      const poster = document.createElement('img');
      poster.src = posterUrl;
      poster.alt = `${movie.title} poster`;
      
      const title = document.createElement('p');
      title.className = 'rec-title';
      title.textContent = movie.title;
      
      recCard.appendChild(poster);
      recCard.appendChild(title);
      
      // Make card clickable
      recCard.addEventListener('click', () => {
        // Close the current modal first
        const currentModal = document.querySelector('.movie-modal');
        if (currentModal) {
          document.body.removeChild(currentModal);
        }
        // Show details for the new movie
        showMovieDetails(movie.id);
      });
      
      recGrid.appendChild(recCard);
    });
    
    container.appendChild(recGrid);
    
  } catch (error) {
    console.error('Error fetching recommendations:', error);
    container.innerHTML = '<p>Failed to load recommendations.</p>';
  }
}

// Get heading text based on endpoint
function getHeadingText(endpoint, query, resultCount) {
  switch (endpoint) {
    case 'search':
      return `Search results for "${query}" (${resultCount})`;
    case 'popular':
      return 'Popular Movies & Shows';
    case 'top-rated':
      return 'Top Rated Movies & Shows';
    case 'random':
      return 'Random Recommendations';
    default:
      return 'Movie Recommendations';
  }
}

// Show/hide loading indicator
function showLoading(isLoading) {
  if (loadingIndicator) {
    loadingIndicator.style.display = isLoading ? 'block' : 'none';
  }
}

// Display error message
function displayError(message) {
  resultsContainer.innerHTML = `
    <div class="error-message">
      <p>${message}</p>
    </div>
  `;
}