from app.dataset import get_results_data
import json

def main() -> None:
    #dataset
    data = get_results_data()
    
    print(json.dumps(data, indent=2))


