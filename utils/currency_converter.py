import requests
from datetime import datetime

class TitanicCurrencyConverter:
    """
    Converts modern earnings to 1912 Titanic-era fare equivalent
    Uses live exchange rates and historical inflation data
    """
    
    def __init__(self):
        # Base inflation multipliers (1912 to 2024)
        self.inflation_multipliers = {
            'USD': 32.0,
            'GBP': 130.0,
            'EUR': 150.0,
            'INR': 1200.0,
        }
        self.base_currency = 'GBP'
        
    def get_live_exchange_rate(self, from_currency, to_currency='GBP'):
        """Fetch live exchange rates"""
        try:
            url = f"https://api.frankfurter.app/latest?from={from_currency}&to={to_currency}"
            response = requests.get(url, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                return data['rates'][to_currency]
            else:
                return self._get_fallback_rate(from_currency, to_currency)
        except Exception:
            return self._get_fallback_rate(from_currency, to_currency)
    
    def _get_fallback_rate(self, from_currency, to_currency):
        """Fallback exchange rates"""
        fallback_rates = {
            'USD': 0.79,
            'EUR': 0.86,
            'INR': 0.0095,
            'GBP': 1.0,
        }
        return fallback_rates.get(from_currency, 1.0)
    
    def modern_to_1912_fare(self, annual_earnings, currency='USD'):
        """Convert modern annual earnings to 1912 fare equivalent"""
        
        if currency != 'GBP':
            exchange_rate = self.get_live_exchange_rate(currency, 'GBP')
            modern_gbp = annual_earnings * exchange_rate
        else:
            modern_gbp = annual_earnings
            exchange_rate = 1.0
        
        inflation_mult = self.inflation_multipliers.get(currency, 100.0)
        equivalent_1912_annual = modern_gbp / inflation_mult
        
        fare_percentage = 0.04
        estimated_fare_1912 = equivalent_1912_annual * fare_percentage
        
        return {
            'fare_1912_gbp': round(estimated_fare_1912, 2),
            'modern_gbp': round(modern_gbp, 2),
            'equivalent_1912_annual_gbp': round(equivalent_1912_annual, 2),
            'currency': currency,
            'exchange_rate': exchange_rate,
            'date_fetched': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }