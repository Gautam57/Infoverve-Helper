Let's explore how to create a machine learning model in Infoveave.

**1.  Creating a New ML Model**

The context describes how to create a new ML Model in Infoveave. 

You can select the required activities from the activity list and arrange them on the ML Models designer canvas to define the workflow.

**2. Selecting a Trained Machine Learning Model**

To select a trained machine learning model for use in the `Execute ML Model` activity, follow these steps:

1. **Locate the Activity Panel:** In the Infoveave interface, you'll find a panel dedicated to activities. This panel lists all available activities, including those related to machine learning.

2. **Navigate to the ML Models Section:** Within the Activity panel, look for a section or category specifically labeled "ML Models" or something similar. This section will house all the trained machine learning models available for use in your workflows.

3. **Select the Desired Model:** Browse through the list of ML Models and click on the one you want to use. The selected model's details, such as its name, description, and type, will be displayed.

4. **Drag and Drop:** Once you've selected the desired ML Model, you can drag and drop it onto the canvas of your workflow. This will add the `Execute ML Model` activity to your workflow, and the selected model will be automatically populated in the configuration fields.

**3. Configuring Infoveave**

To configure Infoveave, you'll need to access the **Configuration panel**.  This panel likely provides options to customize various aspects of the product's settings and behavior. 

Unfortunately, without more specific details about what you want to configure, I can't provide more precise instructions. 

For example, are you trying to:

*  Set up data sources?
*  Define data transformations?
*  Configure user permissions?

Please provide more context about what you'd like to configure, and I'll do my best to guide you through the process.

**4. Linking Data Transformation Activities**

To link your configured data transformation activities with the ML Model in the ML Model builder, you'll need to visually connect them. 

Think of it like building a pipeline:

1. **Identify the Output:** Each data transformation activity has an output node. This is where the transformed data "exits" the activity.

2. **Identify the Input:** The ML Model also has an input node. This is where it expects to receive the prepared data.

3. **Connect the Dots:**  Use your mouse to click and drag a line from the output node of one data transformation activity to the input node of the ML Model. You can repeat this process to link multiple transformation activities in sequence, creating a chain that leads to your ML Model.

This visual connection establishes the flow of data, ensuring that the transformed data from each step is fed into the ML Model for training and prediction.

**5. Validating and Building the Model**

Clicking the "Validate and build" button initiates the process of validating and constructing your ML Model within Infoveave.  This action triggers the model building process. Upon successful completion, the output of the model will be displayed.  

You can then select the model exhibiting the highest accuracy for predicting outcomes using additional input data. 



