import pandas as pd


def build_demo_data():
    suppliers = pd.DataFrame([
        ["Supplier Seattle", 47.6062, -122.3321, 900, 0.92],
        ["Supplier Bay Area", 37.7749, -122.4194, 850, 0.96],
        ["Supplier Los Angeles", 34.0522, -118.2437, 1000, 1.00],
        ["Supplier Denver", 39.7392, -104.9903, 950, 1.02],
        ["Supplier Dallas", 32.7767, -96.7970, 1100, 0.98],
        ["Supplier Chicago", 41.8781, -87.6298, 1050, 1.01],
        ["Supplier Atlanta", 33.7490, -84.3880, 900, 0.99],
        ["Supplier New Jersey", 40.7357, -74.1724, 950, 1.03],
    ], columns=["name", "latitude", "longitude", "capacity", "production_cost"])

    dcs = pd.DataFrame([
        ["DC Seattle", 47.6062, -122.3321, 70000, 900],
        ["DC Sacramento", 38.5816, -121.4944, 62000, 850],
        ["DC Los Angeles", 34.0522, -118.2437, 76000, 1000],
        ["DC Phoenix", 33.4484, -112.0740, 58000, 850],
        ["DC Denver", 39.7392, -104.9903, 61000, 900],
        ["DC Dallas", 32.7767, -96.7970, 72000, 1100],
        ["DC Kansas City", 39.0997, -94.5786, 56000, 850],
        ["DC Chicago", 41.8781, -87.6298, 73000, 1050],
        ["DC Atlanta", 33.7490, -84.3880, 66000, 950],
        ["DC Miami", 25.7617, -80.1918, 60000, 800],
        ["DC New Jersey", 40.7357, -74.1724, 79000, 1000],
        ["DC Boston", 42.3601, -71.0589, 57000, 750],
    ], columns=["name", "latitude", "longitude", "fixed_cost", "capacity"])

    customers = pd.DataFrame([
        ["Seattle", 47.6062, -122.3321, 180],
        ["Portland", 45.5152, -122.6784, 150],
        ["San Francisco", 37.7749, -122.4194, 220],
        ["Los Angeles", 34.0522, -118.2437, 300],
        ["San Diego", 32.7157, -117.1611, 170],
        ["Las Vegas", 36.1699, -115.1398, 120],
        ["Phoenix", 33.4484, -112.0740, 190],
        ["Salt Lake City", 40.7608, -111.8910, 110],
        ["Denver", 39.7392, -104.9903, 180],
        ["Dallas", 32.7767, -96.7970, 230],
        ["Houston", 29.7604, -95.3698, 240],
        ["San Antonio", 29.4241, -98.4936, 140],
        ["Kansas City", 39.0997, -94.5786, 120],
        ["Minneapolis", 44.9778, -93.2650, 140],
        ["Chicago", 41.8781, -87.6298, 260],
        ["Detroit", 42.3314, -83.0458, 140],
        ["Nashville", 36.1627, -86.7816, 130],
        ["Atlanta", 33.7490, -84.3880, 220],
        ["Miami", 25.7617, -80.1918, 190],
        ["Charlotte", 35.2271, -80.8431, 130],
        ["Washington DC", 38.9072, -77.0369, 170],
        ["Philadelphia", 39.9526, -75.1652, 180],
        ["New York", 40.7128, -74.0060, 300],
        ["Boston", 42.3601, -71.0589, 170],
    ], columns=["name", "latitude", "longitude", "demand"])

    return suppliers, dcs, customers
