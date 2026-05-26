import requests
import sys

BASE_URL = "http://localhost:8000/api/v1/movie"


def test_movie(movie_id: str):
    print(f"Testing Movie ID: {movie_id}")
    try:
        aspect_res = requests.get(f"{BASE_URL}/{movie_id}/aspects")

        if aspect_res.status_code == 200:
            data = aspect_res.json()
            print("\n[Aspects Analysis]")

            for aspect, stats in data["aspects"].items():
                print(
                    f'  - {aspect.capitalize()}: Pos {stats["pos"]}%, '
                    f'Neg {stats["neg"]}%, Neu {stats["neu"]}% '
                    f'(Mentions: {stats["mentions"]})'
                )
        else:
            print(f"\n[Aspects Error] Status: {aspect_res.status_code}, " f"Detail: {aspect_res.text}")

        summary_res = requests.get(f"{BASE_URL}/{movie_id}/summary")

        if summary_res.status_code == 200:
            data = summary_res.json()
            print("\n[Summary & Ratings]")
            print(f'  Average User Rating: {data["average_rating"] if data["average_rating"] else "N/A"}')
            print(f'  Generated Summary: {data["summary"]}')
        else:
            print(f"\n[Summary Error] Status: {summary_res.status_code}, " f"Detail: {summary_res.text}")

    except requests.exceptions.ConnectionError:
        print("Error: Cannot connect to FastAPI server. Is it running?")
        sys.exit(1)


if __name__ == "__main__":
    test_id = "435" if len(sys.argv) < 2 else sys.argv[1]
    test_movie(test_id)
