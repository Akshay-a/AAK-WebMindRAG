import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

def get_child_webpages(url):
    """
    Fetches all child webpages (URLs) from the given webpage.

    :param url: The URL of the webpage to scrape.
    :return: A list of child webpage URLs.
    """
    try:
        # Send a GET request to the webpage
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad status codes

        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')
        print(soup.prettify())
        # Find all anchor tags (<a>) with href attributes
        child_webpages = set()  # Use a set to avoid duplicate URLs
        for anchor in soup.find_all('a', href=True):
            # Get the href attribute and join it with the base URL
            child_url = urljoin(url, anchor['href'])
            child_webpages.add(child_url)

        return list(child_webpages)

    except requests.exceptions.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return []

# Example usage
if __name__ == "__main__":
    url = "https://mirror-feeling-d80.notion.site/MCP-From-Scratch-1b9808527b178040b5baf83a991ed3b2"  # Replace with the target website
    child_pages = get_child_webpages(url)
    print(f"Found {len(child_pages)} child webpages:")
    for page in child_pages:
        print(page)