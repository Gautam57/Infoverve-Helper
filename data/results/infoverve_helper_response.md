Let's break down how to create a workflow in Infoveave that uses an API, creates a Datasource, and then performs calculations on the data.

**1. Execute API Activity**

The **Execute API Activity** is your starting point. It allows you to fetch data from any RESTful API. Here's how it works:

*   **Connection:** You'll need to provide the API's URL, the HTTP method (GET, POST, PUT, DELETE), and any necessary headers for authentication or data formatting.
*   **Request Body:** For POST requests, you'll need to define the data you want to send to the API in the request body. This is often in JSON format.
*   **Response Handling:** Infoveave will handle parsing the API's response and making the data accessible within your workflow.

   You can learn more about the Execute API activity here: [https://infoveave-help.pages.dev/automation-v8/activities/execute-api/](https://infoveave-help.pages.dev/automation-v8/activities/execute-api/)

**2. Add a Datasource from the API Response**

Once you've successfully executed the API call, you'll want to create a Datasource from the response data. This will allow you to easily access and manipulate the data within your workflow.

*   **Navigate to Sources:** In the Infoveave interface, go to the "Sources" section.
*   **Add New Source:** Click the "+ Add New Source" button.
*   **Select "Use API":** Choose the "Use API" option to indicate you want to connect to an API.
*   **Name the Source:** Give your Datasource a descriptive name.
*   **Choose Request Method:** Select either "GET" or "POST" depending on the API's requirements.
*   **Enter API URL:**  Input the complete URL of the API endpoint you want to access.
*   **Select Content Type:** Choose the appropriate content type for the API, such as "json" or "form-url-encoded".
*   **Add Headers (if needed):**  If the API requires authentication or other headers, click "Headers" and add them with their corresponding values. Common headers include:
    *   **Authorization:**  Often used for authentication, it might include a token or other credentials.
    *   **Content-Type:** Specifies the format of the data being sent, such as "application/json" for JSON data.
*   **Add Request Body (if needed):** For POST requests, enter the JSON data as the request body.
*   **Run and Test:** Click "Run" to test the API connection and preview the data.
*   **Save:** Once you're satisfied with the connection, click "Save" to create the Datasource.

**3. Add Calculated Columns**

Now that you have your Datasource, you can add calculated columns to perform further analysis and derive new insights from the data.

*   **Navigate to Datasource:** Open the Datasource you created in the "Sources" section.
*   **Click "Add Column":**  Find the option to add a new column to your Datasource.
*   **Choose "Calculated":** Select the "Calculated" option to indicate you want to create a column based on existing data.
*   **Define the Calculation:** Use Infoveave's expression builder to write a formula that calculates the new column's values based on existing columns in your Datasource. You can use mathematical operators, logical functions, and other functions available in Infoveave's expression language.

   You can learn more about adding calculated columns in Infoveave here: [https://infoveave-help.pages.dev/insights-v8/configure-expression/trim/](https://infoveave-help.pages.dev/insights-v8/configure-expression/trim/)



Let me know if you have any more questions or would like to explore specific API examples or calculation scenarios!
