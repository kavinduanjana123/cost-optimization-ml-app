from pulp import LpProblem, LpVariable, LpMinimize, lpSum
import pandas as pd

df = pd.read_excel("unified_company_dataset_with_subscriptions.xlsx")

# Example: optimize total employment cost across departments
departments = df['Department'].unique()

# Decision variable: scaling factor for each department (0.8–1.2 × current)
scale = LpVariable.dicts("scale", departments, lowBound=0.8, upBound=1.2)

# Model
model = LpProblem("Cost_Optimization", LpMinimize)

# Objective: minimize total adjusted cost
model += lpSum([scale[d] * df[df['Department'] == d]['MonthlyTotalEmploymentCost_LKR'].sum()
                for d in departments])

# Constraint example: total scaled cost cannot reduce below 90% of total revenue
model += lpSum([scale[d] * df[df['Department'] == d]['ProjectRevenue_LKR'].sum()
                for d in departments]) >= 0.9 * df['ProjectRevenue_LKR'].sum()

model.solve()

# Output results
print("Optimization Status:", model.status)
for d in departments:
    print(f"{d}: Recommended scaling factor = {scale[d].value():.2f}")
