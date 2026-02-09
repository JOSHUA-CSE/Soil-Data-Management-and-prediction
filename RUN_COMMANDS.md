# Run Commands

## MongoDB
```bash
mongod
```

## Backend (Terminal 1)
```bash
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python manage.py migrate
python manage.py runserver
```

## Frontend (Terminal 2)
```bash
cd frontend
npm install
npm start
```

## Access
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000/api
