import os 
import urllib.request

DATA = [
    {
        "name": "Spain_7370579_S1Hand.tif",
        "url": "https://storage.googleapis.com/sen1floods11/v1.1/data/flood_events/HandLabeled/S1Hand/Spain_7370579_S1Hand.tif",
        "folder": "data/raw/s1"
    },
    {
        "name": "Spain_7370579_S2Hand.tif",
        "url": "https://storage.googleapis.com/sen1floods11/v1.1/data/flood_events/HandLabeled/S2Hand/Spain_7370579_S2Hand.tif",
        "folder": "data/raw/s2"
    },
    {
        "name": "Spain_7370579_LabelHand.tif",
        "url": "https://storage.googleapis.com/sen1floods11/v1.1/data/flood_events/HandLabeled/LabelHand/Spain_7370579_LabelHand.tif",
        "folder": "data/raw/masks"
    }
]


def fetch_data():
    for item in DATA:
        os.makedirs(item["folder"],exist_ok=True)
        out_path = os.path.join(item["folder"],item["name"])

        if not os.path.exists(out_path):
            print(f"Downloading {item['name']}...")
            try:
                urllib.request.urlretrieve(item["url"],out_path)
                print(f"-> Successfully saved to {out_path}")
            except Exception as e:
                print(f"-> Failed to download {item['name']}: {e}")

        else:
            print(f"{item['name']} already exists. Skipping.")
if __name__ == "__main__":
    fetch_data()