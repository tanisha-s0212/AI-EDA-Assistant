# Frontend Setup

## 1. Install dependencies

```bash
cd frontend
npm install
```

## 2. Configure the frontend API target

Create `frontend/.env.local` from `frontend/.env.example`.

```bash
NEXT_PUBLIC_API_BASE_URL=http://localhost:3004/api
```

## 3. Start the frontend

```bash
npm run dev
```

The frontend runs on `http://localhost:3000`.

## Backend reminder

Start the FastAPI backend separately:

```bash
cd ../backend
python -m pip install -r requirements.txt
python main.py
```

The backend runs on `http://localhost:3004`.
