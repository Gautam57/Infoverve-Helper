Let's build that workflow! Here's a breakdown of how to use the 'Execute API' activity to fetch data and then perform calculations within Infoveave:

**1. Setting Up the 'Execute API' Activity**

*   **Purpose:** This activity is your gateway to external data sources. You'll use it to retrieve information from an API and store it in Infoveave.

*   **Configuration:**

    *   **Connection:**  You'll need to establish a connection to your API. Infoveave supports various authentication methods like OAuth, Basic Auth, and API Keys.  Make sure you have the necessary credentials set up.
    *   **Request Type:** Specify the HTTP method you'll use (e.g., GET, POST, PUT, DELETE).
    *   **Request URL:** Enter the complete URL endpoint for the API you want to call. You can use placeholders (like `{{args.param}}`) to dynamically insert values from previous activities.
    *   **Headers:**  If your API requires specific headers (like authorization tokens), add them here.
    *   **Parameters:**  Provide any required query parameters for your API call.
    *   **Body:** If your API requires data to be sent in the request body (e.g., for POST or PUT requests), enter it here in raw JSON or XML format.

*   **Output:** The 'Execute API' activity will return the API's response as JSON data.

**2. Creating a Data Source**

*   **Purpose:**  You'll use the API response to create a new data source in Infoveave. This will allow you to easily access and work with the retrieved data.

*   **Steps:**

    1.  **Navigate to Data Sources:** In Infoveave's interface, find the section for managing data sources.
    2.  **Create New Data Source:** Click the button to create a new data source.
    3.  **Select Data Type:** Choose the appropriate data type for your API response (e.g., JSON).
    4.  **Name Your Data Source:** Give your data source a descriptive name.
    5.  **Configure Ingestion:**

        *   **Data Source Type:** Select the type of data source (e.g., API).
        *   **Connection:** Choose the connection you established for your API.
        *   **Data:**  Select the 'Execute API' activity as the source of your data.

**3. Performing Calculations**

*   **Purpose:** Now that your API data is in Infoveave, you can use various transformation activities to perform calculations.

*   **Activities:**

    *   **Transform Activity (using Javascript):** This activity gives you the flexibility to write custom Javascript code for complex calculations.
    *   **Built-in Calculation Activities:** Infoveave likely has dedicated activities for common calculations like adding, subtracting, multiplying, dividing, rounding, etc.

*   **Example:**

    Let's say your API response contains data about products, including their price and quantity. You want to calculate the total value of each product.

    1.  **Transform Activity:** Use a 'Transform Activity' and write Javascript code to multiply the 'price' and 'quantity' fields for each product.
    2.  **Output:** The transformed data will now include a new field representing the total value.

**Important Notes:**

*   **Error Handling:**  Always consider error handling in your workflow. What happens if the API call fails or the data is not in the expected format?
*   **Data Validation:**  Validate your data after retrieving it from the API to ensure its accuracy and completeness.



Let me know if you have any more questions or want to explore specific calculation examples!
