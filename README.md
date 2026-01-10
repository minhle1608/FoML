Car Price Prediction System
A machine learning-based car price prediction application with a user-friendly interface and RESTful API.
🚀 Quick Start
Follow these steps to set up and run the application:
Step 1: Install Dependencies
bashpip install -r requirements.txt
Step 2: Train the Model
bashpython -m src.train
Note: Run this command from the project root directory
Step 3: Start the API Server
bashuvicorn app.API:app --reload
This will initialize the FastAPI backend server.
Step 4: Launch the Web Interface
bashstreamlit run app/interface.py
📋 Supported Car Brands
For accurate predictions, please use only the following car brands:

Maruti
Skoda
Honda
Hyundai
Toyota
Ford
Renault
Mahindra
Tata
Chevrolet
Fiat
Datsun
Jeep
Mercedes-Benz
Mitsubishi
Audi
Volkswagen
BMW
Nissan
Lexus
Jaguar
Land (Land Rover)
MG
Volvo
Daewoo
Kia
Force
Ambassador
Ashok (Ashok Leyland)
Isuzu
Opel
Peugeot
