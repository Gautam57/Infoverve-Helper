To create a workflow that runs a query from an Oracle DB and sends the results as an email report, you'll use two key activities: **Execute Query** and **Send Email**.

**1. Execute Query:**

This activity allows you to run SQL queries against various data sources, including Oracle databases. 

* **Purpose:**  Fetch data from your Oracle DB by executing SQL queries.
* **Use Cases:**
    * Retrieve real-time data for dashboards or reporting.
    * Modify and reshape data within your workflow.
* **Key Features:**
    * **Data Source Connectivity:** Connect to your Oracle DB using a pre-configured connection.
    * **SQL Query Execution:** Execute your SQL query to retrieve specific data.
    * **Result Handling:** Receive the query results as a tabular output, which you can then use in subsequent activities.

**Configuration:**

* **Connection:** Select the Oracle DB connection you've already set up in Infoveave.
* **Query:** Enter your SQL query. For example, to retrieve all customers from a table named "customers":

```sql
SELECT * FROM customers;
```

* **Output:** Choose how you want to handle the results. You might want to store them in an in-memory table for further processing or save them to a file.

**2. Send Email:**

This activity allows you to compose and send emails as part of your workflow.

* **Dynamic Emailing:** Personalize emails based on the data flowing through your workflow.
* **Templates:** Use pre-designed Infoveave templates for consistent branding.

**Use Cases:**

* Send personalized communications based on incoming data.
* Attach generated reports (like the output from the Execute Query activity) to emails.

**Configuration:**

* **Connection:** Specify the mail server connection to use (either OAuth or SMTP).
* **Use Default Credentials:** If enabled, Infoveave will use its default email credentials. Disable this to use custom authentication.
* **Email To Address, CC Address, BCC Address:** Define static recipient addresses.
* **Subject:** Set the email subject. You can use static text or map it from input data.
* **Content:** Enter the email body. Support is provided for both plain text and HTML formatting.
* **Attachment Name, Sheet Name:** Specify the attachment file name and worksheet to use for formatting the attachment content.
* **Send Data In Email Body:** Include input data as a formatted table within the email body.

**Putting it Together:**

1. **Execute Query:** Run your SQL query against the Oracle DB and store the results.
2. **Send Email:**
   * Set the recipient addresses.
   * Compose the email subject and body.
   * Attach the output from the Execute Query activity as a file.

**Example Workflow:**

Imagine you want to send a monthly sales report to your sales team.

1. **Execute Query:** Retrieve sales data for the current month from your Oracle DB.
2. **Send Email:**
   * Send the email to your sales team's addresses.
   * Set the subject to "Monthly Sales Report - [Month Year]".
   * Include a summary of the sales figures in the email body.
   * Attach the sales data as a CSV file.



