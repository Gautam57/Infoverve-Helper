Let's build that workflow! 

Here's a breakdown of how to use Infoveave to execute an API, create a data source, and perform calculations:

**1.  Execute the API:**

*   **Start with the 'Execute API' activity:** This activity allows you to connect to an external API and retrieve data. You'll need to provide the API endpoint URL, any necessary authentication details, and the parameters for your request. 
*   **Store the response:**  Infoveave will return the API response as data. You can store this data in a variable for later use.

**2. Create a Data Source:**

*   **Use the 'Update Datasource' activity:** This activity lets you create a new data source from the data you retrieved from the API. 
*   **Specify the data source type:** Choose the appropriate data source type based on the format of your API response (e.g., JSON, CSV).
*   **Name your data source:** Give your data source a descriptive name for easy identification.

**3. Perform Calculations:**

*   **Access your data source:**  Use the 'Query Datasource' activity to access the data you stored in your newly created data source.
*   **Apply calculations:** Infoveave offers a wide range of calculation functions. You can use these functions to perform arithmetic operations, aggregate data, filter data, and more.

**Example Workflow:**

1.  **Execute API:**
    *   API Endpoint: `https://api.example.com/data`
    *   Authentication: API Key (stored as a secret in Infoveave)
    *   Parameters: `?limit=100`
    *   Store response in variable: `api_response`

2.  **Update Datasource:**
    *   Data Source Type: JSON
    *   Data Source Name: `API_Data`
    *   Data Source Content: `api_response`

3.  **Query Datasource:**
    *   Data Source: `API_Data`
    *   Query: `SELECT * FROM data`

4.  **Calculate:**
    *   Use the 'Calculate' activity to perform calculations on the data retrieved from the API. For example, you could calculate the average value of a specific column.

**Remember:**

*   Replace the example API endpoint and parameters with your own.
*   Refer to the Infoveave documentation for a complete list of available activities and functions.



Let me know if you have any other questions or need help with a specific part of the workflow! 
