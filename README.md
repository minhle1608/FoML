Step to run this:
Step 1: run pip install -r requirements.txt
Step 2: run python -m src.train (from the current folder imported)
Step 3: run uvicorn app.API:app --reload (initialize API)
Step 4: run streamlit run app/interface.py 
For the prediction, please only input these brand:
['Maruti', 'Skoda', 'Honda', 'Hyundai', 'Toyota', 'Ford', 'Renault', 'Mahindra',
 'Tata' ,'Chevrolet', 'Fiat', 'Datsun', 'Jeep', 'Mercedes-Benz', 'Mitsubishi',
 'Audi', 'Volkswagen', 'BMW', 'Nissan', 'Lexus', 'Jaguar', 'Land', 'MG', 'Volvo',
 'Daewoo', 'Kia', 'Force', 'Ambassador', 'Ashok', 'Isuzu', 'Opel', 'Peugeot']



