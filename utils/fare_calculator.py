import requests
from datetime import datetime

class ModernFareCalculator:
    """Convert 1912 Titanic fares to modern equivalent prices"""
    
    def __init__(self):
        self.inflation_multipliers = {
            'USD': 32.0,
            'GBP': 130.0,
            'EUR': 150.0,
            'INR': 1200.0,
        }
        
        self.price_comparisons = {
            'ranges': [
                (0, 50, [
                    "a Spotify Premium subscription for 5 months",
                    "two large pizzas with delivery",
                    "a basic haircut at a salon"
                ]),
                (50, 150, [
                    "a one-way domestic flight from NYC to Miami",
                    "a pair of Nike Air Max sneakers",
                    "dinner for two at a nice steakhouse"
                ]),
                (150, 400, [
                    "a round-trip flight from Los Angeles to Las Vegas",
                    "an Apple Watch SE",
                    "a night at a 4-star hotel in downtown Chicago"
                ]),
                (400, 800, [
                    "a one-way flight from New York to London (economy)",
                    "an Xbox Series X with 2 games",
                    "a weekend Airbnb in Miami Beach"
                ]),
                (800, 1500, [
                    "a round-trip flight from San Francisco to Tokyo (economy)",
                    "a Samsung Galaxy S24 Ultra",
                    "front-row concert tickets for Taylor Swift (2 tickets)"
                ]),
                (1500, 3000, [
                    "a round-trip business class flight from NYC to Paris",
                    "a MacBook Pro 14-inch",
                    "a week-long all-inclusive resort in Cancun for two"
                ]),
                (3000, 6000, [
                    "a round-trip business class flight from LA to Dubai",
                    "a used Honda Civic (down payment)",
                    "a 7-day Mediterranean cruise for two"
                ]),
                (6000, 12000, [
                    "first class round-trip tickets from New York to Singapore",
                    "a fully loaded MacBook Pro 16-inch with accessories",
                    "two months of rent in Manhattan (studio apartment)"
                ]),
                (12000, 25000, [
                    "a decent used Toyota Camry or Honda Accord",
                    "a year of college tuition at a state university",
                    "a two-week luxury safari in Kenya for two people"
                ]),
                (25000, float('inf'), [
                    "a brand new Tesla Model 3",
                    "a year's worth of first-class international travel",
                    "a down payment on a condo in Austin, Texas"
                ])
            ]
        }
    
    def get_live_exchange_rate(self, from_currency='GBP', to_currency='USD'):
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
        """Fallback rates"""
        rates = {
            ('GBP', 'USD'): 1.27,
            ('GBP', 'EUR'): 1.17,
            ('GBP', 'INR'): 105.0,
            ('GBP', 'GBP'): 1.0,
            ('USD', 'GBP'): 0.79,
            ('EUR', 'GBP'): 0.86,
            ('INR', 'GBP'): 0.0095,
            ('USD', 'USD'): 1.0,
            ('EUR', 'EUR'): 1.0,
            ('INR', 'INR'): 1.0,
        }
        return rates.get((from_currency, to_currency), 1.0)
    
    def fare_1912_to_modern(self, fare_1912_gbp, original_currency='USD'):
        """Convert 1912 fare to modern equivalent in user's currency"""
        modern_gbp = fare_1912_gbp * self.inflation_multipliers['GBP']
        
        if original_currency != 'GBP':
            exchange_rate = self.get_live_exchange_rate('GBP', original_currency)
            modern_value_original = modern_gbp * exchange_rate
        else:
            exchange_rate = 1.0
            modern_value_original = modern_gbp
        
        if original_currency != 'USD':
            usd_rate = self.get_live_exchange_rate(original_currency, 'USD')
            usd_value = modern_value_original * usd_rate
        else:
            usd_value = modern_value_original
        
        comparisons = self._get_price_comparisons(usd_value)
        
        return {
            'fare_1912_gbp': round(fare_1912_gbp, 2),
            'modern_value_original_currency': round(modern_value_original, 2),
            'original_currency': original_currency,
            'modern_gbp': round(modern_gbp, 2),
            'modern_usd': round(usd_value, 2),
            'exchange_rate_to_original': exchange_rate,
            'comparisons': comparisons,
            'date_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def _get_price_comparisons(self, usd_amount):
        """Get contextual price comparisons"""
        for min_val, max_val, comparisons in self.price_comparisons['ranges']:
            if min_val <= usd_amount < max_val:
                return comparisons
        return ["a significant luxury purchase"]