# Ask-Tomorrowland

## Overview
Tomorrowland Explorer is a Python application designed to scrape web pages, process textual data, and provide answers to questions about the contents of the webpages via Retrieval Augmented Generation (RAG) combined with an LLM. It leverages Django for its web framework and includes various features for data validation and user interaction, including a citing of sources used to produce a response.

## Key Features
1. **Web Scraping**: Scrape web pages to gather links and content.
2. **Database Creation**: Build a local or MongoDB database from the scraped data.
3. **Query Processing**: Perform queries on the database to retrieve relevant information.
4. **Django Integration**: Use Django to create a web interface for interacting with the application.

## Installation & Usage

### Prerequisites
- Python 3.8+
- Recommended: Virtual environment (e.g., `venv`)

Install dependencies:
`pip install -r requirements.txt`

Navigate to the project directory:
`cd proj`

Set up environment variables: Create a .env file in the project root

### Usage
Run the Django development server:
`python manage.py runserver`
Access the application in your web browser at http://127.0.0.1:8000/.

Home Page: Access the home page at /.
Scrape Webpages: Start the web scraping process at /scrape.
Create Database: Build the database from the scraped data at /make_db.
Query: Perform queries on the database at /index.

### File Structure
Modules
1. views.py
Handles web requests and responses.
Includes endpoints for home, scraping webpages, creating a database, and performing queries.
2. services.py
Contains the core logic for scraping, database creation, LLM initialization, and query performance.

### Extending the Application
To extend the application, you can add new views, models, and services as needed. Ensure to update the URLs and templates accordingly.

1. Utilize the json response obtained by replicating the requests that occur when loading  https://winter.tomorrowland.com/en/line-up/?page=timetable&day=2025-03-21 to augment the available website content data.
2. Perform evals using the attached human generated questions and Ask-Tomorrowland responses - when is the model performing well and when has it missed the mark? Identify (ir-)relevant sources to quantify precision and recall.
3. Improve non-"store" site parsing of HTML contents. Content that is not related to merchandise does not seem to be optimally scraped. Refer to metrics from 2 to measure improvements/regressions.
4. Experiment with document chunking/segmentation - refer again to 2.
5. Test the langchain webloader to simplify the scraping process
6. Write (LLM-assisted?) tests


Author
Daniel Wolff