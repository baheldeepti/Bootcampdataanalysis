📚 Measuring the True Impact of Coding Bootcamps: A Causal Inference Approach
🛠 Project Overview
This project estimates the true causal effect of attending a coding bootcamp on annual salaries using causal inference techniques.

Rather than relying on naive correlations, we simulate a realistic observational dataset with confounders and apply Propensity Score Matching (PSM) to adjust for selection bias.

📈 Problem Statement
While coding bootcamps advertise significant salary boosts, observational data often suffers from selection bias — individuals self-select into bootcamps based on characteristics like education and experience.

This project aims to answer:

"What is the true average impact of attending a coding bootcamp on salary, after adjusting for confounders?"

🔎 Dataset
Synthetic dataset of 5,000 individuals with features:

age

prior_experience

education_level

region

treatment (bootcamp attended = 1 / not attended = 0)

salary

Confounders are intentionally embedded into treatment assignment.

🧠 Methodology
Data Simulation: Realistic generation of observational data with biased treatment assignment

Exploratory Data Analysis (EDA): Visualize selection bias across confounders

Naive Estimation: Initial salary difference without adjustment

Propensity Score Matching (PSM):

Estimate propensity scores using logistic regression

Match treated and untreated individuals based on nearest neighbor matching

Causal Effect Estimation: Calculate the true Average Treatment Effect on the Treated (ATT)

📊 Key Results

Step	Result
Naive Salary Difference	+$13,757
True Causal Effect (ATT after PSM)	+$10,003
📚 Conclusion
Coding bootcamps do causally increase salaries by approximately $10,000, after adjusting for confounding factors.

This project highlights the necessity of causal inference in understanding real-world effects and avoiding misleading conclusions based on raw correlations.

🚀 Future Work
Apply Difference-in-Differences (DiD) if pre/post bootcamp salary data becomes available

Use Causal Forests to identify heterogeneous treatment effects (who benefits the most)

Incorporate additional covariates (technical skills, certifications, portfolio strength)

💻 Tech Stack
Python (pandas, numpy, seaborn, matplotlib, sklearn)

Jupyter Notebook / Google Colab

📎 Repository Structure
bash
Copy
Edit
├── data/
│   └── causal_inference_bootcamp_dataset.csv
├── notebooks/
│   └── causal_inference_bootcamp_analysis.ipynb
├── README.md
👤 Author
Deepti Bahel
Data Analyst | Data Scientist | Causal Inference Enthusiast
