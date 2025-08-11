Let's break down how to create a workflow in Infoveave to execute your Oracle DB query and send the results via email every morning.

**1. The `Execute Query` Activity**

The `Execute Query` activity is your gateway to retrieving data from your Oracle database. Here's how to use it:

* **Connection:** First, you'll need to establish a connection to your Oracle database within Infoveave. This involves providing connection details like the database server address, port, username, and password.

* **Query:**  This is where you input your SQL query.  For example, if you want to retrieve customer names and their latest order dates, your query might look like this:

   ```sql
   SELECT customer_name, MAX(order_date) AS latest_order
   FROM customers
   JOIN orders ON customers.customer_id = orders.customer_id
   GROUP BY customer_name;
   ```

* **Output:** The `Execute Query` activity will return the results of your query as a tabular dataset. This data will be available for use in subsequent activities within your workflow.

**2. The `Send Email` Activity**

Now, let's configure the `Send Email` activity to deliver the query results to your customer:

* **Connection:** Choose the email server connection you want to use. Infoveave supports various options like OAuth connections for Gmail or Outlook, or SMTP connections for other email providers.

* **Recipient:** Specify the email address(es) of your customer(s). You can use a static email address or dynamically pull it from your input data.

* **Subject:** Craft a clear and concise subject line for your email.

* **Content:**  This is where you'll format the email body. You can use plain text or HTML.  Here's an example using HTML to present the query results in a table:

   ```html
   <h2>Customer Order Summary</h2>
   <table>
     <thead>
       <tr>
         <th>Customer Name</th>
         <th>Latest Order Date</th>
       </tr>
     </thead>
     <tbody>
       <!--  Infoveave will dynamically insert the query results here -->
     </tbody>
   </table>
   ```

* **Attachments:** If you need to include additional files with your email, you can attach them here.

**Putting it Together: Your Automated Workflow**

1. **Start:** Your workflow begins with a trigger. For daily execution, you'd use a "Schedule" trigger set to run every morning at your desired time.

2. **Execute Query:** The workflow then executes your `Execute Query` activity, retrieving the data from your Oracle database.

3. **Send Email:** Finally, the `Send Email` activity sends the email to your customer, dynamically populating the email body with the query results.

**Example Workflow**

Let's say you want to send a daily email with the top 5 selling products from your database.

1. **Trigger:** Schedule trigger to run daily at 8:00 AM.
2. **Execute Query:**
   * Connection: Oracle database connection.
   * Query: 
     ```sql
     SELECT product_name, SUM(quantity_sold) AS total_sold
     FROM sales
     GROUP BY product_name
     ORDER BY total_sold DESC
     LIMIT 5;
     ```
3. **Send Email:**
   * Recipient: `customer@example.com`
   * Subject: `Daily Top Selling Products`
   * Content: HTML template with a table displaying the product name and total sold.

**Important Notes:**

* **Error Handling:**  Consider adding error handling to your workflow to gracefully handle any issues with the database connection or email sending.
* **Security:**  Always store sensitive database credentials securely and avoid hardcoding them directly into your workflow.


