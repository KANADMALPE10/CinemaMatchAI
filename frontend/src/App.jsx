import { useCallback, useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import "./App.css";

const API_BASE = (import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000").replace(/\/$/, "");

const TMDB_IMAGE = "https://image.tmdb.org/t/p/w500";
const TMDB_BACKDROP = "https://image.tmdb.org/t/p/w1280";

const GENRES = [
  "All",
  "Action",
  "Adventure",
  "Animation",
  "Comedy",
  "Crime",
  "Documentary",
  "Drama",
  "Fantasy",
  "Horror",
  "Mystery",
  "Romance",
  "Science Fiction",
  "Thriller",
];

function posterUrl(path) {
  if (!path) return null;

  if (path.startsWith("http")) return path;

  return `${TMDB_IMAGE}${path}`;
}

function backdropUrl(path) {
  if (!path) return null;

  if (path.startsWith("http")) return path;

  return `${TMDB_BACKDROP}${path}`;
}

function normalizeGenre(value) {
  if (!value) return [];

  if (Array.isArray(value)) return value;

  const text = String(value).trim();

  if (!text) return [];

  if (text.includes("|")) {
    return text.split("|").map((item) => item.trim()).filter(Boolean);
  }

  if (text.includes(",")) {
    return text.split(",").map((item) => item.trim()).filter(Boolean);
  }

  const known = [
    "Science Fiction",
    "TV Movie",
    "Action",
    "Adventure",
    "Animation",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Family",
    "Fantasy",
    "History",
    "Horror",
    "Music",
    "Mystery",
    "Romance",
    "Thriller",
    "War",
    "Western",
  ];

  const result = [];
  let remaining = text;

  for (const genre of known.sort((a, b) => b.length - a.length)) {
    const pattern = new RegExp(
      `(?<!\\w)${genre.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}(?!\\w)`,
      "i",
    );

    if (pattern.test(remaining)) {
      result.push(genre);
      remaining = remaining.replace(pattern, " ");
    }
  }

  return [...new Set([...result, ...remaining.split(/\s+/).filter(Boolean)])];
}

async function fetchJson(url, signal) {
  const response = await fetch(url, { signal });

  if (!response.ok) {
    let message = `Request failed (${response.status})`;

    try {
      const data = await response.json();
      message = data.detail || message;
    } catch {
      // Use the fallback error message.
    }

    throw new Error(message);
  }

  return response.json();
}

function Star({ filled = true }) {
  return (
    <svg
      width="14"
      height="14"
      viewBox="0 0 24 24"
      fill={filled ? "currentColor" : "none"}
      stroke="currentColor"
      strokeWidth="1.8"
      aria-hidden="true"
    >
      <path d="m12 2.5 2.9 5.9 6.5.9-4.7 4.6 1.1 6.5-5.8-3.1-5.8 3.1 1.1-6.5-4.7-4.6 6.5-.9L12 2.5Z" />
    </svg>
  );
}

function SearchIcon() {
  return (
    <svg
      width="19"
      height="19"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      aria-hidden="true"
    >
      <circle cx="10.8" cy="10.8" r="6.8" />
      <path d="m16 16 4.5 4.5" />
    </svg>
  );
}

function PlayIcon() {
  return (
    <svg
      width="16"
      height="16"
      viewBox="0 0 24 24"
      fill="currentColor"
      aria-hidden="true"
    >
      <path d="M7 4.8c0-.8.9-1.3 1.6-.9l11 6.2c.8.4.8 1.4 0 1.8l-11 6.2c-.7.4-1.6-.1-1.6-.9V4.8Z" />
    </svg>
  );
}

function Logo({ onClick }) {
  return (
    <a
      href="#home"
      className="brand-home-link"
      aria-label="CinemaMatch AI home"
      onClick={(event) => {
        event.preventDefault();
        onClick?.();
      }}
    >
      <img
        src="/images/cinematch-logo.png"
        alt="CinemaMatch AI"
        className="brand-logo"
      />
    </a>
  );
}

function SectionHeading({ title, subtitle, action, onAction }) {
  return (
    <div className="section-heading">
      <div>
        <h2>{title}</h2>
        {subtitle && <p>{subtitle}</p>}
      </div>

      {action && (
        <button className="text-action" onClick={onAction}>
          {action}
          <span aria-hidden="true"> ↗</span>
        </button>
      )}
    </div>
  );
}

function MovieCard({ movie, onClick, featured = false }) {
  const poster = posterUrl(movie.poster_path);
  const rating = Number(movie.rating ?? movie.vote_average ?? 0);

  return (
    <button
      className={`movie-card ${featured ? "movie-card-featured" : ""}`}
      onClick={() => onClick(movie)}
      aria-label={`View details for ${movie.title}`}
    >
      <div className="poster-frame">
        {poster ? (
          <img
            className="movie-poster"
            src={poster}
            alt={`${movie.title} poster`}
            loading="lazy"
          />
        ) : (
          <div className="poster-fallback">
            <span>CM</span>
            <p>{movie.title}</p>
          </div>
        )}

        <div className="poster-overlay">
          <span className="poster-play">
            <PlayIcon />
          </span>
          <span className="poster-open-label">View details</span>
        </div>

        {movie.similarity != null && (
          <span className="match-label">
            {Math.round(Number(movie.similarity) * 100)}% MATCH
          </span>
        )}
      </div>

      <div className="movie-card-info">
        <h3 title={movie.title}>{movie.title}</h3>

        <div className="movie-card-meta">
          <span>{movie.year || "Release year unavailable"}</span>

          {rating > 0 && (
            <span className="rating">
              <Star />
              {rating.toFixed(1)}
            </span>
          )}
        </div>
      </div>
    </button>
  );
}

function MovieCarousel({
  title,
  subtitle,
  movies,
  onMovieClick,
  loading = false,
  error = "",
  emptyMessage = "Movies will appear here when available.",
  onRetry,
}) {
  const trackRef = useRef(null);

  function scroll(direction) {
    if (!trackRef.current) return;

    trackRef.current.scrollBy({
      left: direction * Math.max(trackRef.current.clientWidth * 0.75, 300),
      behavior: "smooth",
    });
  }

  return (
    <section className="movie-section">
      <SectionHeading title={title} subtitle={subtitle} />

      {loading ? (
        <div className="loading-row">
          {Array.from({ length: 6 }).map((_, index) => (
            <div className="skeleton-card" key={index}>
              <div className="skeleton-poster" />
              <div className="skeleton-line" />
              <div className="skeleton-line skeleton-line-short" />
            </div>
          ))}
        </div>
      ) : error ? (
        <div className="inline-error">
          <p>{error}</p>
          {onRetry && (
            <button className="button-secondary" onClick={onRetry}>
              Try again
            </button>
          )}
        </div>
      ) : movies.length === 0 ? (
        <div className="empty-row">{emptyMessage}</div>
      ) : (
        <div className="carousel-wrap">
          <button
            className="carousel-arrow carousel-arrow-left"
            onClick={() => scroll(-1)}
            aria-label={`Scroll ${title} left`}
          >
            ‹
          </button>

          <div className="movie-track" ref={trackRef}>
            {movies.map((movie) => (
              <MovieCard
                key={movie.tmdb_id}
                movie={movie}
                onClick={onMovieClick}
              />
            ))}
          </div>

          <button
            className="carousel-arrow carousel-arrow-right"
            onClick={() => scroll(1)}
            aria-label={`Scroll ${title} right`}
          >
            ›
          </button>
        </div>
      )}
    </section>
  );
}

function MovieModal({ movie, onClose, onMovieClick }) {
  const [recommendations, setRecommendations] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    if (!movie?.tmdb_id) return undefined;

    const controller = new AbortController();

    setRecommendations([]);
    setError("");
    setLoading(true);

    fetchJson(
      `${API_BASE}/api/movies/${movie.tmdb_id}/similar?limit=12`,
      controller.signal,
    )
      .then((data) => {
        setRecommendations(data.results || []);
      })
      .catch((err) => {
        if (err.name !== "AbortError") {
          setError(
            "We couldn't load recommendations. Please try again.",
          );
        }
      })
      .finally(() => {
        if (!controller.signal.aborted) {
          setLoading(false);
        }
      });

    return () => controller.abort();
  }, [movie?.tmdb_id]);

  useEffect(() => {
    const previousOverflow = document.body.style.overflow;

    document.body.style.overflow = "hidden";

    function onKeyDown(event) {
      if (event.key === "Escape") onClose();
    }

    window.addEventListener("keydown", onKeyDown);

    return () => {
      document.body.style.overflow = previousOverflow;
      window.removeEventListener("keydown", onKeyDown);
    };
  }, [onClose]);

  if (!movie) return null;

  const poster = posterUrl(movie.poster_path);
  const backdrop = backdropUrl(movie.backdrop_path);
  const genres = normalizeGenre(movie.genres);
  const rating = Number(movie.rating ?? movie.vote_average ?? 0);

  return createPortal(
    <div
      className="modal-backdrop"
      role="presentation"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) onClose();
      }}
    >
      <section
        className="movie-modal"
        role="dialog"
        aria-modal="true"
        aria-label={`${movie.title} details`}
      >
        <div className="modal-hero">
          {backdrop && (
            <img
              className="modal-backdrop-image"
              src={backdrop}
              alt=""
            />
          )}

          <div className="modal-hero-gradient" />

          <button
            className="modal-close"
            onClick={onClose}
            aria-label="Close movie details"
          >
            ×
          </button>

          <div className="modal-hero-content">
            <span className="eyebrow">MOVIE DETAILS</span>
            <h2>{movie.title}</h2>

            <div className="modal-meta">
              {movie.year && <span>{movie.year}</span>}

              {rating > 0 && (
                <span className="rating">
                  <Star />
                  {rating.toFixed(1)}
                </span>
              )}

              {movie.vote_count > 0 && (
                <span>
                  {Number(movie.vote_count).toLocaleString()} votes
                </span>
              )}
            </div>
          </div>
        </div>

        <div className="modal-body">
          <div className="modal-overview-grid">
            <div className="modal-poster-wrap">
              {poster ? (
                <img src={poster} alt={`${movie.title} poster`} />
              ) : (
                <div className="poster-fallback">{movie.title}</div>
              )}
            </div>

            <div className="modal-description">
              <span className="eyebrow">THE STORY</span>

              <p>
                {movie.overview ||
                  "A description for this movie is not available in the catalog."}
              </p>

              {genres.length > 0 && (
                <div className="modal-genres">
                  {genres.map((genre) => (
                    <span className="genre-tag" key={genre}>
                      {genre}
                    </span>
                  ))}
                </div>
              )}
            </div>
          </div>

          <div className="recommendation-heading">
            <div>
              <span className="eyebrow">PICKED FOR YOU</span>
              <h3>Because you watched {movie.title}</h3>
              <p>Discover movies with similar stories and genres.</p>
            </div>
          </div>

          {loading ? (
            <div className="recommendation-loading">
              <span className="spinner" />
              Finding your next favorite movie...
            </div>
          ) : error ? (
            <div className="inline-error">
              <p>{error}</p>
              <button
                className="button-secondary"
                onClick={() => {
                  setLoading(true);
                  setError("");

                  fetchJson(
                    `${API_BASE}/api/movies/${movie.tmdb_id}/similar?limit=12`,
                  )
                    .then((data) => {
                      setRecommendations(data.results || []);
                    })
                    .catch(() => {
                      setError("Recommendations are currently unavailable.");
                    })
                    .finally(() => setLoading(false));
                }}
              >
                Try again
              </button>
            </div>
          ) : recommendations.length > 0 ? (
            <div className="modal-recommendation-track">
              {recommendations.map((item) => (
                <MovieCard
                  key={item.tmdb_id}
                  movie={item}
                  onClick={onMovieClick}
                />
              ))}
            </div>
          ) : (
            <div className="empty-row">
              No similar movies were found for this title.
            </div>
          )}
        </div>
      </section>
    </div>,
    document.body,
  );
}

function App() {
  const [movies, setMovies] = useState([]);
  const [topRated, setTopRated] = useState([]);
  const [genreMovies, setGenreMovies] = useState([]);
  const [searchResults, setSearchResults] = useState([]);

  const [search, setSearch] = useState(
    () => new URLSearchParams(window.location.search).get("q") || "",
  );
  const [activeGenre, setActiveGenre] = useState("All");
  const [selectedMovie, setSelectedMovie] = useState(null);

  const [loading, setLoading] = useState(true);
  const [topLoading, setTopLoading] = useState(true);
  const [genreLoading, setGenreLoading] = useState(true);
  const [searchLoading, setSearchLoading] = useState(false);

  const [homeError, setHomeError] = useState("");
  const [topError, setTopError] = useState("");
  const [genreError, setGenreError] = useState("");
  const [searchError, setSearchError] = useState("");

  const handleSearchChange = useCallback((value) => {
    const nextSearch = value;
    const currentSearch = new URLSearchParams(window.location.search).get("q") || "";
    const nextUrl = new URL(window.location.href);

    if (!currentSearch && nextSearch.trim()) {
      nextUrl.searchParams.set("q", nextSearch);
      nextUrl.hash = "";
      window.history.pushState({ cinemaMatchSearch: true }, "", nextUrl);
    } else if (currentSearch && nextSearch.trim()) {
      nextUrl.searchParams.set("q", nextSearch);
      nextUrl.hash = "";
      window.history.replaceState({ cinemaMatchSearch: true }, "", nextUrl);
    } else if (!nextSearch.trim() && currentSearch) {
      nextUrl.searchParams.delete("q");
      nextUrl.hash = "home";
      window.history.pushState({ cinemaMatchHome: true }, "", nextUrl);
    }

    setSearch(nextSearch);
  }, []);

  const navigateHome = useCallback(() => {
    const nextUrl = new URL(window.location.href);
    nextUrl.searchParams.delete("q");
    nextUrl.hash = "home";

    if (new URLSearchParams(window.location.search).has("q")) {
      window.history.pushState({ cinemaMatchHome: true }, "", nextUrl);
    } else {
      window.history.pushState({ cinemaMatchHome: true }, "", nextUrl);
    }

    setSearch("");
    setSelectedMovie(null);
    setActiveGenre("All");
    window.scrollTo({ top: 0, behavior: "smooth" });
  }, []);

  useEffect(() => {
    function handlePopState() {
      const params = new URLSearchParams(window.location.search);
      setSearch(params.get("q") || "");
      setSelectedMovie(null);
    }

    window.addEventListener("popstate", handlePopState);
    return () => window.removeEventListener("popstate", handlePopState);
  }, []);

  const loadHome = useCallback(async () => {
    const controller = new AbortController();

    setLoading(true);
    setHomeError("");

    try {
      const data = await fetchJson(
        `${API_BASE}/api/movies?limit=60`,
        controller.signal,
      );

      const results = data.results || [];

      setMovies(results);
    } catch (error) {
      if (error.name !== "AbortError") {
        setHomeError(
          "We couldn't connect to the movie catalog. Make sure the FastAPI backend is running.",
        );
      }
    } finally {
      setLoading(false);
    }
  }, []);

  const loadTopRated = useCallback(async () => {
    setTopLoading(true);
    setTopError("");

    try {
      const data = await fetchJson(
        `${API_BASE}/api/movies?limit=100`,
      );

      const results = (data.results || []).sort(
        (a, b) => Number(b.rating || 0) - Number(a.rating || 0),
      );

      setTopRated(results.slice(0, 24));
    } catch {
      setTopError("Unable to load top-rated movies.");
    } finally {
      setTopLoading(false);
    }
  }, []);

  const loadGenreMovies = useCallback(async () => {
    setGenreLoading(true);
    setGenreError("");

    try {
      const data = await fetchJson(
        `${API_BASE}/api/movies?limit=100`,
      );

      setGenreMovies(data.results || []);
    } catch {
      setGenreError("Unable to load genre collections.");
    } finally {
      setGenreLoading(false);
    }
  }, []);

  useEffect(() => {
    loadHome();
    loadTopRated();
    loadGenreMovies();
  }, [loadHome, loadTopRated, loadGenreMovies]);

  useEffect(() => {
    const query = search.trim();

    if (!query) {
      setSearchResults([]);
      setSearchError("");
      setSearchLoading(false);
      return undefined;
    }

    const controller = new AbortController();

    const timer = window.setTimeout(async () => {
      setSearchLoading(true);
      setSearchError("");

      try {
        const data = await fetchJson(
          `${API_BASE}/api/movies?q=${encodeURIComponent(query)}&limit=30`,
          controller.signal,
        );

        setSearchResults(data.results || []);
      } catch (error) {
        if (error.name !== "AbortError") {
          setSearchError("Search is unavailable. Check the backend.");
        }
      } finally {
        if (!controller.signal.aborted) {
          setSearchLoading(false);
        }
      }
    }, 300);

    return () => {
      window.clearTimeout(timer);
      controller.abort();
    };
  }, [search]);

  const openMovie = useCallback((movie) => {
    setSelectedMovie(movie);
  }, []);

  const closeMovie = useCallback(() => {
    setSelectedMovie(null);
  }, []);

  const filteredMovies =
    activeGenre === "All"
      ? movies
      : movies.filter((movie) =>
          normalizeGenre(movie.genres).some(
            (genre) => genre.toLowerCase() === activeGenre.toLowerCase(),
          ),
        );

  const filteredGenreMovies =
    activeGenre === "All"
      ? genreMovies
      : genreMovies.filter((movie) =>
          normalizeGenre(movie.genres).some(
            (genre) => genre.toLowerCase() === activeGenre.toLowerCase(),
          ),
        );

  const heroMovie =
    movies.find((movie) => movie.backdrop_path && movie.overview) ||
    movies.find((movie) => movie.poster_path && movie.overview) ||
    movies[0];

  return (
    <div className="app-shell" id="home">
      <header className="topbar">
        <Logo onClick={navigateHome} />

        <nav className="main-nav" aria-label="Main navigation">
          <a className="nav-link nav-link-active" href="#home">
            Discover
          </a>
          <a className="nav-link" href="#trending">
            Trending
          </a>
          <a className="nav-link" href="#top-rated">
            Top Rated
          </a>
        </nav>

        <div className="header-search">
          <SearchIcon />

          <input
            type="search"
            placeholder="Search movies, genres..."
            value={search}
            onChange={(event) => handleSearchChange(event.target.value)}
            aria-label="Search movies"
          />

          <kbd>⌘ K</kbd>
        </div>
      </header>

      <main>
        {search.trim() ? (
          <div className="search-page page-container">
            <div className="search-page-heading">
              <span className="eyebrow">MOVIE SEARCH</span>
              <h1>
                Discover <span>something new.</span>
              </h1>
              <p>
                Results for <strong>“{search.trim()}”</strong>
              </p>
            </div>

            {searchLoading ? (
              <div className="search-status">
                <span className="spinner" />
                Searching the movie catalog...
              </div>
            ) : searchError ? (
              <div className="inline-error">{searchError}</div>
            ) : searchResults.length > 0 ? (
              <div className="search-results-grid">
                {searchResults.map((movie) => (
                  <MovieCard
                    key={movie.tmdb_id}
                    movie={movie}
                    onClick={openMovie}
                  />
                ))}
              </div>
            ) : (
              <div className="empty-row search-empty">
                <h3>No movies found</h3>
                <p>Try a different title or search term.</p>
              </div>
            )}
          </div>
        ) : (
          <>
            <section className="hero-section">
              {heroMovie?.backdrop_path && (
                <img
                  className="hero-backdrop"
                  src={backdropUrl(heroMovie.backdrop_path)}
                  alt=""
                  fetchPriority="high"
                />
              )}

              <div className="hero-scrim" />

              <div className="hero-content page-container">
                <div className="hero-copy">
                  <div className="hero-badge">
                    <span className="hero-badge-dot" />
                    YOUR PERSONAL MOVIE DISCOVERY ENGINE
                  </div>

                  <h1>
                    Find films that
                    <br />
                    <span>feel like you.</span>
                  </h1>

                  <p>
                    Not sure what to watch next? Discover your next
                    favorite film through stories, genres, and
                    recommendations made to match your taste.
                  </p>

                  <div className="hero-buttons">
                    <a className="button-primary" href="#trending">
                      <PlayIcon />
                      Explore movies
                    </a>

                    {heroMovie && (
                      <button
                        className="button-glass"
                        onClick={() => openMovie(heroMovie)}
                      >
                        Discover featured film <span>↗</span>
                      </button>
                    )}
                  </div>

                  <div className="hero-trust">
                    <span className="trust-symbol">✦</span>
                    <span>Powered by content-based AI recommendations</span>
                  </div>
                </div>

                {heroMovie && (
                  <button
                    className="hero-feature-card"
                    onClick={() => openMovie(heroMovie)}
                    aria-label={`View ${heroMovie.title}`}
                  >
                    {posterUrl(heroMovie.poster_path) ? (
                      <img
                        src={posterUrl(heroMovie.poster_path)}
                        alt={`${heroMovie.title} poster`}
                      />
                    ) : (
                      <div className="poster-fallback">
                        {heroMovie.title}
                      </div>
                    )}

                    <div className="hero-feature-overlay">
                      <span className="eyebrow">FEATURED FILM</span>
                      <h2>{heroMovie.title}</h2>
                      <div className="hero-feature-meta">
                        <span>{heroMovie.year || "Movie"}</span>
                        {Number(heroMovie.rating) > 0 && (
                          <span className="rating">
                            <Star />
                            {Number(heroMovie.rating).toFixed(1)}
                          </span>
                        )}
                      </div>
                    </div>
                  </button>
                )}
              </div>

              <div className="hero-bottom-line" />
            </section>

            <div className="page-container main-page-content">
              <section className="discovery-intro">
                <div>
                  <span className="eyebrow">YOUR NEXT MOVIE NIGHT</span>
                  <h2>What are you in the mood for?</h2>
                </div>

                <p>
                  Explore stories worth your time, all in one place.
                </p>
              </section>

              <section className="genre-section">
                <div className="genre-pills" aria-label="Filter by genre">
                  {GENRES.map((genre) => (
                    <button
                      key={genre}
                      className={`genre-pill ${
                        activeGenre === genre ? "genre-pill-active" : ""
                      }`}
                      onClick={() => setActiveGenre(genre)}
                    >
                      {genre}
                    </button>
                  ))}
                </div>
              </section>

              <div id="trending">
                <MovieCarousel
                  title={
                    activeGenre === "All"
                      ? "Trending on CinemaMatch"
                      : `${activeGenre} Movies`
                  }
                  subtitle="Popular films to add to your watchlist"
                  movies={filteredMovies}
                  loading={loading}
                  error={homeError}
                  onRetry={loadHome}
                  onMovieClick={openMovie}
                  emptyMessage="No movies found for this genre."
                />
              </div>

              <MovieCarousel
                title="Top Rated"
                subtitle="Highly rated picks from the catalog"
                movies={topRated}
                loading={topLoading}
                error={topError}
                onRetry={loadTopRated}
                onMovieClick={openMovie}
              />

              <section className="discovery-banner">
                <div className="banner-orb banner-orb-one" />
                <div className="banner-orb banner-orb-two" />

                <div className="banner-copy">
                  <span className="eyebrow">SMARTER DISCOVERY</span>
                  <h2>
                    One movie leads to
                    <br />
                    <span>another great story.</span>
                  </h2>
                  <p>
                    Open any movie to discover films with similar
                    stories and genres.
                  </p>
                </div>

                <a href="#trending" className="button-primary">
                  Find your next film <span>↗</span>
                </a>
              </section>

              <MovieCarousel
                title="Explore More"
                subtitle="More movies, more possibilities"
                movies={filteredGenreMovies}
                loading={genreLoading}
                error={genreError}
                onRetry={loadGenreMovies}
                onMovieClick={openMovie}
              />

              <footer className="site-footer">
                <Logo onClick={navigateHome} />

                <p>
                  Discover stories you'll love. Built with curiosity
                  and AI.
                </p>

                <span>© {new Date().getFullYear()} CinemaMatch AI</span>
              </footer>
            </div>
          </>
        )}
      </main>

      {selectedMovie && (
        <MovieModal
          key={selectedMovie.tmdb_id}
          movie={selectedMovie}
          onClose={closeMovie}
          onMovieClick={openMovie}
        />
      )}
    </div>
  );
}

export default App;