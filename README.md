# 🎵 LyriMatch

> **Discover music through the power of lyrics.** LyriMatch uses advanced AI embeddings to match your favorite lyrics with similar songs, helping you find your next musical obsession.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![React](https://img.shields.io/badge/React-18.3+-61dafb.svg)](https://reactjs.org/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.5+-3178c6.svg)](https://www.typescriptlang.org/)
[![Flask](https://img.shields.io/badge/Flask-Latest-black.svg)](https://flask.palletsprojects.com/)

---

## 📋 Table of Contents

- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [Running the Application](#-running-the-application)
- [API Endpoints](#-api-endpoints)
- [Project Structure](#-project-structure)
- [How It Works](#-how-it-works)
- [Experiments & Evaluation](#-experiments--evaluation)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [License](#-license)

---

## ✨ Features

### 🔍 **Lyric-Based Search**
- Search for songs using any lyrics or phrases
- Get top 5 most similar songs based on semantic meaning
- Powered by sentence transformers and FAISS vector similarity

### 📝 **Add Songs Manually**
- Input song title and full lyrics
- Automatically generates and stores embeddings
- Duplicate detection prevents redundant entries

### 🎼 **Spotify Playlist Import**
- Paste any Spotify playlist URL
- Automatic lyrics fetching from Genius
- Real-time progress tracking with visual feedback
- Smart error handling for missing lyrics
- Batch processing with individual song status

### 📊 **Progress Tracking**
- Live status updates during playlist imports
- Color-coded indicators:
  - ✅ **Green**: Successfully added
  - ⚠️ **Yellow**: Skipped (duplicate)
  - ❌ **Red**: Failed (lyrics not found)
  - 🔵 **Blue**: Currently processing
  - ⚪ **Gray**: Pending
- Detailed error messages per song

### 🛡️ **Robust Error Handling**
- Automatic text truncation for long lyrics
- Duplicate song detection
- Graceful handling of API failures
- Individual song failures don't stop batch processing

---

## 🛠️ Tech Stack

### Backend
- **Flask**: Lightweight Python web framework
- **Sentence Transformers**: `all-MiniLM-L6-v2` model for text embeddings
- **FAISS**: Facebook AI Similarity Search for fast vector operations
- **Spotipy**: Spotify Web API Python library
- **LyricsGenius**: Genius API wrapper for lyrics fetching
- **PyArrow & Pandas**: Efficient data storage and manipulation

### Frontend
- **React 18**: Modern UI library with hooks
- **TypeScript**: Type-safe JavaScript
- **Vite**: Lightning-fast build tool
- **Tailwind CSS**: Utility-first CSS framework
- **Framer Motion**: Smooth animations
- **Lucide React**: Beautiful icon library
- **React Router**: Client-side routing

---

## 📦 Prerequisites

Before you begin, ensure you have the following installed:

| Requirement | Version | Download |
|------------|---------|----------|
| **Python** | 3.8 or higher | [python.org](https://www.python.org/downloads/) |
| **Node.js** | 16.x or higher | [nodejs.org](https://nodejs.org/) |
| **npm** | 8.x or higher | Comes with Node.js |
| **Git** | Latest | [git-scm.com](https://git-scm.com/) |

### API Keys Required

You'll need API credentials from these services (all free tiers available):

1. **Spotify Developer Account**
   - Create at: [developer.spotify.com](https://developer.spotify.com)
   - Needed for: Fetching playlist information

2. **Genius API Account**
   - Create at: [genius.com/api-clients](https://genius.com/api-clients)
   - Needed for: Fetching song lyrics

---

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/FroostySnoowman/LyriMatch.git
cd LyriMatch
```

### 2. Backend Setup

#### Create Virtual Environment

```bash
cd backend
python3 -m venv venv
```

#### Activate Virtual Environment

**macOS/Linux:**
```bash
source venv/bin/activate
```

**Windows (Command Prompt):**
```cmd
venv\Scripts\activate
```

**Windows (PowerShell):**
```powershell
venv\Scripts\Activate.ps1
```

#### Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> ⚠️ **Note**: First-time installation may take 5-10 minutes as it downloads the ML models.

### 3. Frontend Setup

Open a **new terminal** window/tab:

```bash
cd frontend
npm install
```

---

## ⚙️ Configuration

### Backend Configuration

1. Navigate to the `backend` directory:
   ```bash
   cd backend
   ```

2. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```

3. Edit `.env` and add your API credentials:
   ```env
   SPOTIFY_CLIENT_ID=your_spotify_client_id_here
   SPOTIFY_CLIENT_SECRET=your_spotify_client_secret_here
   GENIUS_API_KEY=your_genius_api_key_here
   ```

#### 🔑 Getting Your API Keys

**Spotify API Keys:**
1. Go to [Spotify Developer Dashboard](https://developer.spotify.com/dashboard)
2. Click "Create an App"
3. Fill in app name and description
4. Copy the **Client ID** and **Client Secret**

**Genius API Key:**
1. Go to [Genius API Clients](https://genius.com/api-clients)
2. Click "New API Client"
3. Fill in app details (redirect URI can be `http://localhost`)
4. Generate a **Client Access Token**
5. Copy the token

### Frontend Configuration

1. Navigate to the `frontend` directory:
   ```bash
   cd frontend
   ```

2. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```

3. Edit `.env` (defaults should work for local development):
   ```env
   VITE_API_BASE_URL=http://127.0.0.1:8080
   VITE_SUPABASE_URL=your-supabase-url (optional)
   VITE_SUPABASE_ANON_KEY=your-supabase-anon-key (optional)
   ```

> 💡 **Note**: Supabase configuration is optional and only needed if you're using authentication features.

---

## 🏃 Running the Application

### Method 1: Using Separate Terminals (Recommended)

#### Terminal 1 - Start Backend

```bash
cd backend
source venv/bin/activate  # or venv\Scripts\activate on Windows
python app.py
```

You should see:
```
Loading embeddings from: data\song_embeddings\songembeddings.parquet
Loaded X songs from parquet
Initialization complete. Songs loaded: X, Embedding dim: 384
 * Running on http://127.0.0.1:8080
```

#### Terminal 2 - Start Frontend

```bash
cd frontend
npm run dev
```

You should see:
```
  VITE v5.4.2  ready in XXX ms

  ➜  Local:   http://localhost:5173/
  ➜  Network: use --host to expose
```

### Method 2: Using the Setup Script

For Unix-based systems (macOS/Linux):

```bash
chmod +x setup.sh
./setup.sh
```

> ⚠️ **Note**: This script only starts the backend. You'll still need to run the frontend separately.

### Access the Application

Open your browser and navigate to:
```
http://localhost:5173
```

The frontend will automatically proxy API requests to the backend at `http://127.0.0.1:8080`.

---

## 📡 API Endpoints

### Base URL
```
http://127.0.0.1:8080
```

### Endpoints

#### 🔍 Search for Similar Songs

```http
POST /search
Content-Type: application/json

{
  "lyrics": "I've been living a lonely life"
}
```

**Response:**
```json
{
  "results": [
    {
      "rank": 1,
      "name": "Ho Hey - The Lumineers",
      "album_name": "",
      "id": 42,
      "cosine_sim": 0.87
    }
  ]
}
```

#### ➕ Add Song Manually

```http
POST /add_song
Content-Type: application/json

{
  "title": "Song Title",
  "lyrics": "Full lyrics here..."
}
```

**Response:**
```json
{
  "status": "ok",
  "id": 123,
  "title": "Song Title"
}
```

#### 📋 Get Playlist Tracks

```http
POST /get_playlist_tracks
Content-Type: application/json

{
  "playlist_url": "https://open.spotify.com/playlist/...",
  "song_limit": 10  // optional
}
```

**Response:**
```json
{
  "status": "ok",
  "tracks": [
    {
      "title": "Song Name",
      "artists": ["Artist Name"]
    }
  ],
  "total": 10
}
```

#### 🎵 Add Song from Search

```http
POST /add_song_from_search
Content-Type: application/json

{
  "title": "Song Title",
  "artist": "Artist Name"
}
```

**Response:**
```json
{
  "status": "ok",
  "id": 124,
  "title": "Song Title",
  "artist": "Artist Name"
}
```

#### 🏥 Health Check

```http
GET /health
```

**Response:**
```json
{
  "status": "ok",
  "index_loaded": true,
  "model_loaded": true,
  "song_count": 150
}
```

### Error Responses

| Status Code | Description |
|------------|-------------|
| `200` | Success |
| `400` | Bad Request (missing parameters) |
| `404` | Not Found (lyrics not found) |
| `409` | Conflict (duplicate song) |
| `500` | Server Error |

---

## 📁 Project Structure

```text
LyriMatch/
├── backend/
│   ├── app.py                             # Flask API server
│   ├── build_lyrics_dataset_from_playlist.py  # Build evaluation CSV from Spotify + Genius
│   ├── experiments_tfidf_vs_embedding.py      # Offline TF-IDF vs embedding evaluation
│   ├── lyrics.py                        # Lyrics processing utilities
│   ├── requirements.txt                 # Python dependencies
│   ├── .env.example                     # Environment variables template
│   ├── data/
│   │   ├── song_embeddings/             # Stored embeddings for the live API
│   │   │   └── songembeddings.parquet
│   │   └── lyrics_dataset.csv           # Offline evaluation dataset (built via script)
│   └── venv/                            # Virtual environment (git-ignored)
│
├── frontend/
│   ├── src/
│   │   ├── components/                  # Reusable UI components
│   │   ├── pages/                       # Page components
│   │   │   ├── App.tsx                 # Main search page
│   │   │   ├── AddSong.tsx             # Add songs/playlists
│   │   │   ├── Landing.tsx             # Landing page
│   │   │   └── ...
│   │   ├── lib/
│   │   │   ├── api.ts                  # API client functions
│   │   │   └── supabase.ts             # Supabase client (optional)
│   │   └── main.tsx                    # App entry point
│   ├── package.json                    # Node dependencies
│   ├── .env.example                    # Environment variables template
│   ├── vite.config.ts                  # Vite configuration
│   └── tailwind.config.js              # Tailwind CSS config
│
├── setup.sh                            # Automated setup script (Unix)
├── .gitignore                          # Git ignore rules
└── README.md                           # This file
```

---

## 🧠 How It Works

### 1. **Text Embedding**
- Uses the `all-MiniLM-L6-v2` sentence transformer model
- Converts lyrics into 384-dimensional vectors
- Captures semantic meaning, not just keywords

### 2. **Vector Similarity Search**
- FAISS (Facebook AI Similarity Search) indexes all song embeddings
- Cosine similarity measures how "close" two songs are
- Returns top-k most similar results in milliseconds

### 3. **Data Storage**
- Embeddings stored in Apache Parquet format
- Efficient columnar storage with compression
- Fast loading and querying with pandas

### 4. **Lyrics Fetching**
- Spotify API provides track metadata
- Genius API retrieves full lyrics
- Automatic retry and error handling

### 5. **Duplicate Prevention**
- Case-insensitive song name checking
- Prevents redundant database entries
- Returns HTTP 409 for duplicates

### 6. **Text Truncation**
- Long lyrics automatically truncated to 512 tokens
- Prevents tensor dimension mismatches
- Maintains model compatibility

---

## 🧪 Experiments & Evaluation

LyriMatch includes a small offline evaluation pipeline to compare a **classical TF-IDF baseline** with the **embedding-based model** used in the live API.

These scripts are intended for **reproducible experiments** (e.g., course project reports) and do **not** affect the running Flask server.

### 1. Build the Evaluation Dataset

Use a Spotify playlist + Genius lyrics to build `data/lyrics_dataset.csv`:

```bash
cd backend
source venv/bin/activate
python build_lyrics_dataset_from_playlist.py
```

You will be prompted for:

- A Spotify playlist URL
- An optional maximum number of songs to pull

For each track in the playlist:

- Spotify is used to get the title and artist
- Genius is used to fetch full lyrics
- Successful entries are stored in `data/lyrics_dataset.csv` with columns:

```text
song_id, title, artist, lyrics
```

You can run this script multiple times on different playlists; new songs will be appended.

### 2. Run TF-IDF vs Embedding Experiments

Once `lyrics_dataset.csv` exists, run:

```bash
cd backend
source venv/bin/activate   # if not already active
python experiments_tfidf_vs_embedding.py
```

This script will:

1. Load `data/lyrics_dataset.csv`.
2. Build a **TF-IDF baseline** (unigrams + bigrams, 50k vocabulary).
3. Build an **embedding matrix** using `all-MiniLM-L6-v2` (same model as the API) and a FAISS index.
4. Construct an evaluation set of songs where each artist has ≥ 4 songs.
5. Define "relevant" songs as **other tracks by the same artist**.
6. Compute **Precision@5** and **Recall@5** for both models.
7. Print a few **qualitative examples** (query + top-5 neighbors from each method) that can be copied into reports/slides.

Example output (abridged):

```text
=== Quantitative results ===
TF-IDF baseline:     P@5 = 0.088, R@5 = 0.022
Embedding + FAISS:   P@5 = 0.076, R@5 = 0.017

=== Qualitative examples ===

Example 1: Query song
  ID:    316
  Title: Sober
  Artist: P!nk

  TF-IDF top-5 neighbors:
    1. Sober — P!nk
    ...

  Embedding top-5 neighbors:
    1. Sober — P!nk
    2. 1-800-273-8255 — Logic
    3. Without Me — Halsey
    ...
```

You can use these numbers and examples directly in your **Methods**, **Results**, and **Discussion** sections to show:

- How a classical TF-IDF baseline compares to a modern embedding model.
- That you have a clear evaluation protocol (P@k / R@k).
- Concrete examples of how each model behaves.

---

## 🐛 Troubleshooting

### Common Issues

#### **Backend won't start**

**Problem**: `ModuleNotFoundError: No module named 'flask'`

**Solution**:
```bash
cd backend
source venv/bin/activate  # Ensure venv is activated
pip install -r requirements.txt
```

---

#### **Port already in use**

**Problem**: `Address already in use: 127.0.0.1:8080`

**Solution**:
```bash
# Find process using port 8080
lsof -i :8080  # macOS/Linux
netstat -ano | findstr :8080  # Windows

# Kill the process or change port in app.py
```

---

#### **API keys not working**

**Problem**: `401 Unauthorized` or `403 Forbidden`

**Solution**:
1. Verify API keys are correctly copied (no extra spaces)
2. Check that `.env` file is in the correct directory
3. Restart the backend after changing `.env`
4. Ensure API keys are valid and not expired

---

#### **Lyrics not found**

**Problem**: Many songs showing as "failed" during playlist import

**Solution**:
- This is normal! Not all songs have lyrics on Genius
- Instrumental tracks will always fail
- Some newer songs may not be indexed yet
- Live versions often have different titles

---

#### **Frontend can't connect to backend**

**Problem**: `Network Error` or `CORS error`

**Solution**:
1. Verify backend is running on port 8080
2. Check `VITE_API_BASE_URL` in `frontend/.env`
3. Ensure CORS is enabled in Flask (it should be)
4. Try accessing `http://127.0.0.1:8080/health` directly

---

#### **Slow embedding generation**

**Problem**: Adding songs takes a long time

**Solution**:
- First run downloads the ML model (~100MB)
- Subsequent runs are much faster
- CPU-only processing is slower than GPU
- Consider reducing playlist size for testing

---

### Platform-Specific Issues

#### **macOS**

If you encounter SSL certificate errors:
```bash
/Applications/Python\ 3.x/Install\ Certificates.command
```

#### **Windows**

If you get execution policy errors in PowerShell:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

#### **Linux**

May need to install additional dependencies:
```bash
sudo apt-get update
sudo apt-get install python3-dev python3-pip
```

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/AmazingFeature`
3. **Commit your changes**: `git commit -m 'Add some AmazingFeature'`
4. **Push to the branch**: `git push origin feature/AmazingFeature`
5. **Open a Pull Request**

### Development Guidelines

- Follow existing code style
- Add comments for complex logic
- Test thoroughly before submitting
- Update documentation if needed

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Sentence Transformers** - For the embedding model
- **FAISS** - For fast similarity search
- **Spotify** - For playlist and track metadata
- **Genius** - For lyrics data
- **The Open Source Community** - For amazing tools and libraries

---

## 📧 Contact & Support

Having issues? Here's how to get help:

1. **Check the [Troubleshooting](#-troubleshooting) section**
2. **Search existing [GitHub Issues](https://github.com/yourusername/LyriMatch/issues)**
3. **Open a new issue** with:
   - Your OS and Python/Node versions
   - Error messages and logs
   - Steps to reproduce

---

<div align="center">

**Made with ❤️ and 🎵**

Star ⭐ this repository if you find it helpful!

</div>