import requests
import datetime
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor


class MINDS:
    def __init__(self):
        self.CLINICAL_URL = "https://portal.gdc.cancer.gov/auth/api/clinical_tar"
        self.BIOSPECIMEN_URL = "https://portal.gdc.cancer.gov/auth/api/biospecimen_tar"
        self.session = requests.Session()

    def build_data(self):
        """Build and return the data payload for the POST request."""
        size = 100_000  # TODO: get this number from the GDC API
        filters = '{"op":"and","content":[{"op":"in","content":{"field":"files.access","value":["open"]}}]}'
        data = {
            "size": size,
            "attachment": True,
            "format": "TSV",
            "filters": filters,
        }
        return data

    def download(self, url):
        """
        Download data from the specified URL and display a progress bar.
        """
        data = self.build_data()
        today = datetime.datetime.today().strftime("%Y-%m-%d")

        if url == self.CLINICAL_URL:
            data_type = "clinical"
        elif url == self.BIOSPECIMEN_URL:
            data_type = "biospecimen"

        try:
            response = self.session.post(url, data=data, stream=True)
            response.raise_for_status()

            file_size = int(response.headers.get("Content-Length", 0))
            chunk_size = 1024
            progress_bar = tqdm(total=file_size, unit="B", unit_scale=True)

            with open(f"{data_type}.cases_selection.{today}.tar.gz", "wb") as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    progress_bar.update(len(chunk))
                    f.write(chunk)
            progress_bar.close()
            print("File downloaded successfully.")
        except requests.exceptions.RequestException as e:
            print("Failed to download the file:", e)

    def update(self):
        """
        Update and download both Clinical and Biospecimen data.
        """
        print("Downloading Clinical Data")
        self.download(self.CLINICAL_URL)
        print("Clinical Download Complete")
        print("-" * 50)
        print("Downloading Biospecimen Data")
        self.download(self.BIOSPECIMEN_URL)
        print("Biospecimen Download Complete")


def main():
    minds = MINDS()
    with ThreadPoolExecutor() as executor:
        executor.submit(minds.update)


if __name__ == "__main__":
    main()
